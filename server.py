"""
PyTorch Practice Backend Server
- Code execution with real PyTorch
- AI evaluation via DeepSeek/Bedrock
- Authentication with allowlist
"""
import os
import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import boto3
from botocore.config import Config
import base64

# ============== CONFIG ==============
USERS = {
    "hawkoli1987": "Matajinqiu2!",
}

AWS_REGION = "us-east-1"
BEDROCK_MODEL_ID = "deepseek.v3.2"

app = FastAPI(title="PyTorch Practice API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== MODELS ==============
class ExecuteRequest(BaseModel):
    code: str
    test_code: str | None = None
    expected_output: str | None = None
    expected_pattern: str | None = None
    expected_shape: str | None = None


class ExecuteResponse(BaseModel):
    output: str
    error: str | None = None
    test_passed: bool = False
    expected_output: str | None = None


class EvaluateRequest(BaseModel):
    question: str
    user_answer: str | None = None
    user_code: str | None = None
    user_output: str | None = None
    expected_output: str | None = None
    code_hint: str | None = None
    type: str = "explanation"  # "coding" or "explanation"
    test_passed: bool = False


class EvaluateResponse(BaseModel):
    correct: bool
    feedback: str


# ============== AUTH ==============
def verify_auth(authorization: str | None = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    if not authorization.startswith("Basic "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    try:
        encoded = authorization.replace("Basic ", "")
        decoded = base64.b64decode(encoded).decode("utf-8")
        username, password = decoded.split(":", 1)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if username not in USERS or USERS[username] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return username


# ============== API ROUTES ==============

@app.post("/api/execute", response_model=ExecuteResponse)
async def execute_code(request: ExecuteRequest):
    """Execute Python code with optional unit test verification"""
    import sys
    import io
    
    output = ""
    error = None
    test_passed = False
    expected_output = request.expected_output or ""
    
    # Pre-import common libraries
    preimport_code = """
import torch
import numpy as np
"""
    
    # Execution context with pre-imported modules
    exec_globals = {
        "__builtins__": __builtins__,
        "torch": torch,
        "np": np,
    }
    exec_locals = {}
    
    try:
        # Run pre-import
        exec(preimport_code, exec_globals, exec_locals)
        
        code = request.code.strip()
        
        # Capture stdout
        old_stdout = sys.stdout
        redirected = io.StringIO()
        
        # Handle multi-line code - execute all lines, display last expression result
        lines = code.split('\n')
        
        try:
            # Try to execute as single expression first
            compiled = compile(code, '<string>', 'eval')
            sys.stdout = redirected
            result = eval(code, exec_globals, exec_locals)
            sys.stdout = old_stdout
            redirected.truncate(0)
            redirected.seek(0)
            # Format result like Jupyter
            if hasattr(result, 'pretty_print'):
                output = result.pretty_print()
            elif hasattr(result, '__repr__'):
                output = repr(result)
            else:
                output = str(result)
                
        except (SyntaxError, ValueError):
            # It's statements - execute normally and capture any output
            sys.stdout = redirected
            
            # For blocks like 'with', 'for', 'if', we need to execute the whole thing
            # Try to execute the whole code block
            try:
                # Check if it's a single line with semicolons
                if ';' in code and '\n' not in code.strip():
                    # Execute as single statement
                    exec(code, exec_globals, exec_locals)
                    sys.stdout = old_stdout
                    output = redirected.getvalue()
                elif '\n' in code.strip():
                    # Multi-line - could have blocks
                    # Execute the whole thing and get the last expression's result
                    exec(code, exec_globals, exec_locals)
                    sys.stdout = old_stdout
                    output = redirected.getvalue()
                    
                    # If output is empty but we have a last expression value,
                    # try to get it from exec_locals or the last line
                    if not output.strip():
                        # Try to evaluate the last line as expression
                        last_line = [l.strip() for l in lines if l.strip()][-1] if lines else ''
                        if last_line and not any(last_line.startswith(kw) for kw in ['if', 'for', 'while', 'with', 'def', 'class']):
                            try:
                                result = eval(last_line, exec_globals, exec_locals)
                                if hasattr(result, '__repr__'):
                                    output = repr(result)
                            except:
                                pass
                else:
                    # Single line statement
                    exec(code, exec_globals, exec_locals)
                    sys.stdout = old_stdout
                    output = redirected.getvalue()
            except Exception as exec_err:
                sys.stdout = old_stdout
                redirected.truncate(0)
                redirected.seek(0)
                stdout_output = redirected.getvalue()
                if not stdout_output.strip() and output == "":
                    raise exec_err
        
        # If there's a test code, run it
        test_error = None
        if request.test_code:
            redirected_test = io.StringIO()
            old_stdout_test = sys.stdout
            sys.stdout = redirected_test
            
            try:
                exec(request.test_code, exec_globals, exec_locals)
                sys.stdout = old_stdout_test
                test_output = redirected_test.getvalue()
                
                # Test passed if:
                # 1. expected_output matches in test_output, OR
                # 2. expected_output empty and test ran without error
                if request.expected_output:
                    # Check if expected is in test output (normalize whitespace)
                    norm_expected = ' '.join(request.expected_output.split())
                    norm_test = ' '.join(test_output.split())
                    if norm_expected in norm_test or norm_test == norm_expected:
                        test_passed = True
                    # If not matching, test_passed stays False - will trigger LLM
                else:
                    # No expected_output - if test ran without error, pass
                    test_passed = True
                    # Store test output for reference
                    if test_output.strip():
                        output = test_output.strip()
                    
            except AssertionError as ae:
                sys.stdout = old_stdout_test
                test_passed = False
                test_error = f"Assertion failed: {str(ae)}"
                output += f"\n[Test Error] {test_error}"
            except Exception as test_e:
                sys.stdout = old_stdout_test
                test_passed = False
                test_error = str(test_e)
                output += f"\n[Test Error] {test_error}"
        
        # Verify output against expected if no test_code
        if not test_passed and request.expected_output and not request.test_code:
            # Compare actual output with expected (normalize whitespace)
            normalized_output = ' '.join(output.split())
            normalized_expected = ' '.join(request.expected_output.split())
            if normalized_expected in normalized_output or normalized_output == normalized_expected:
                test_passed = True
        
        # Check expected pattern (shape, etc)
        if not test_passed and request.expected_pattern:
            if request.expected_pattern.lower() in output.lower():
                test_passed = True
        
        # Check expected shape
        if not test_passed and request.expected_shape:
            if f"({request.expected_shape})" in output or f"[{request.expected_shape}]" in output:
                test_passed = True
        
    except Exception as e:
        sys.stdout = old_stdout
        error = str(e)
    
    return ExecuteResponse(
        output=output, 
        error=error,
        test_passed=test_passed,
        expected_output=expected_output if not test_passed else None
    )


@app.post("/api/evaluate", response_model=EvaluateResponse)
async def evaluate_answer(
    request: EvaluateRequest,
    user: str = Depends(verify_auth)
):
    """Evaluate using DeepSeek via Bedrock"""
    
    client = boto3.client(
        'bedrock-runtime',
        region_name=AWS_REGION,
        config=Config(signature_version='bearer')
    )
    
    # Build prompt based on type
    if request.type == "coding":
        system_prompt = f"""You are a helpful PyTorch coding tutor. The user attempted a coding problem and their submission has FAILED the unit test.

## TASK:
Question: {request.question}
Hint: {request.code_hint or 'No hint provided'}

## USER'S SUBMISSION:
User's code:
```python
{request.user_code}
```

User's output:
```
{request.user_output}
```

Expected output/pattern:
{request.expected_output or request.code_hint or 'See question above'}

## WHAT HAPPENED:
- Unit test status: FAILED

## YOUR TASK:
1. First, explain WHAT went wrong - is it a syntax error, wrong output, wrong approach?
2. Explain WHY the user's approach is incorrect
3. Give the CORRECT solution with full working code example
4. Explain the key concept being tested

Be SPECIFIC and HELPFUL. Don't just say "incorrect" - explain the concept they misunderstood.

## RESPONSE FORMAT:
{{"correct": false, "feedback": "YOUR DETAILED EXPLANATION INCLUDING THE CORRECT ANSWER"}}

IMPORTANT: Your feedback MUST include the correct answer code at the end."""""
    else:
        system_prompt = f"""You are a helpful PyTorch tutor evaluating a student's explanation.

## TASK:
Question: {request.question}

## STUDENT'S ANSWER:
{request.user_answer}

## EVALUATION CRITERIA:
- Check if the student CAPTURED THE GIST (key concepts) - be LENIENT on exact wording
- Minor omissions or imprecise language should NOT mark wrong if the core concept is correct
- If the gist is captured but incomplete: mark CORRECT but provide complementary info

## RESPONSE FORMAT:
{{"correct": boolean, "feedback": "YOUR CONCISE FEEDBACK"}}

## FEEDBACK STYLE:
- Be encouraging and concise
- If CORRECT: briefly confirm, then add any complementary insights
- If WRONG (missed gist): explain what's missing clearly
- Keep feedback focused - no long essays

Examples of good feedback:
- "✅ Correct! Just to add: torch.tensor() creates a copy while torch.from_numpy() shares memory with the numpy array."
- "⚠️ You got the main idea about gradients, but missing: zero_grad() clears gradients BEFORE the backward pass to avoid accumulation across batches."
"""

    try:
        response = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Please evaluate this answer."}
                ],
                "max_tokens": 800,
                "temperature": 0.7
            })
        )
        
        response_body = json.loads(response["body"].read())
        content = response_body["choices"][0]["message"]["content"]
        
        # Extract JSON
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            result = json.loads(match.group())
            return EvaluateResponse(**result)
        else:
            return EvaluateResponse(
                correct=False,
                feedback="Could not parse evaluation. " + content[:200]
            )
        
    except Exception as e:
        return EvaluateResponse(
            correct=False,
            feedback=f"Evaluation error: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "pytorch_version": torch.__version__,
        "users": len(USERS)
    }


# ============== RUN ==============
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 PyTorch Practice Backend")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Users: {len(USERS)}")
    print("\n   Endpoints:")
    print("   - POST /api/execute  (code execution + testing)")
    print("   - POST /api/evaluate (AI evaluation)")
    print("   - GET  /api/health   (health check)")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
