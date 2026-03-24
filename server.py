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
        
        # Check if code is a single expression (Jupyter-style display)
        code = request.code.strip()
        
        # Capture stdout
        old_stdout = sys.stdout
        redirected = io.StringIO()
        sys.stdout = redirected
        
        # Try to compile as expression first
        try:
            compiled = compile(code, '<string>', 'eval')
            # It's an expression - evaluate and display result
            result = eval(code, exec_globals, exec_locals)
            sys.stdout = old_stdout
            redirected.truncate(0)
            redirected.seek(0)
            # Format result like Jupyter - use repr for tensors
            if hasattr(result, 'pretty_print'):
                output = result.pretty_print()
            elif hasattr(result, '__repr__'):
                output = repr(result)
            else:
                output = str(result)
        except SyntaxError:
            # It's a statement - execute normally
            sys.stdout = redirected
            exec(code, exec_globals, exec_locals)
            sys.stdout = old_stdout
            output = redirected.getvalue()
        except Exception as eval_err:
            sys.stdout = old_stdout
            redirected.truncate(0)
            redirected.seek(0)
            output = redirected.getvalue()
            if not output:
                raise eval_err
        
        # If there's a test code, run it
        if request.test_code:
            redirected_test = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = redirected_test
            
            try:
                exec(request.test_code, exec_globals, exec_locals)
                sys.stdout = old_stdout
                test_output = redirected_test.getvalue()
                
                # Test passed if no error and output matches expected
                if request.expected_output and request.expected_output in test_output:
                    test_passed = True
                elif not request.expected_output:
                    test_passed = True
                    
            except Exception as test_e:
                sys.stdout = old_stdout
                test_passed = False
        
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
        system_prompt = f"""You are a PyTorch coding tutor. The user was asked to solve this coding problem:

Question: {request.question}
Hint: {request.code_hint or 'No hint provided'}

User's code:
{request.user_code}

User's output:
{request.user_output}

Expected output/pattern:
{request.expected_output or request.code_hint or 'See question above'}

Compare the user's solution with the expected solution. Provide brief, constructive feedback if incorrect.

Respond in JSON format:
{{"correct": boolean, "feedback": "brief explanation"}}
"""
    else:
        system_prompt = f"""You are a PyTorch expert tutor. Evaluate the user's explanation of this concept:

Question: {request.question}

User's Answer: {request.user_answer}

Provide encouraging feedback with corrections if needed. Focus on accuracy and completeness.

Respond in JSON format:
{{"correct": boolean, "feedback": "detailed explanation"}}
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
