"""
PyTorch Practice Backend Server
- Code execution with real PyTorch
- AI evaluation via DeepSeek/Bedrock
- Authentication with allowlist
"""
import os
import json
import asyncio
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
# User allowlist: username -> password
USERS = {
    "hawkoli1987": "Matajinqiu2!",
}

# AWS Bedrock config
AWS_REGION = "us-east-1"
BEDROCK_MODEL_ID = "deepseek.v3.2"

# Bedrock credentials (from ~/.ssh/.bedrock)
AWS_BEARER_TOKEN = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")

app = FastAPI(title="PyTorch Practice API")

# Enable CORS for GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== MODELS ==============
class ExecuteRequest(BaseModel):
    code: str


class ExecuteResponse(BaseModel):
    output: str
    error: str | None = None


class EvaluateRequest(BaseModel):
    question: str
    user_answer: str
    code_hint: str | None = None


class EvaluateResponse(BaseModel):
    score: int
    correct: bool
    feedback: str


# ============== AUTH ==============
def verify_auth(authorization: str | None = Header(None)) -> str:
    """Verify Basic Auth credentials"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    if not authorization.startswith("Basic "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    # Decode credentials
    try:
        encoded = authorization.replace("Basic ", "")
        decoded = base64.b64decode(encoded).decode("utf-8")
        username, password = decoded.split(":", 1)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check allowlist
    if username not in USERS or USERS[username] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return username


# ============== API ROUTES ==============

@app.post("/api/execute", response_model=ExecuteResponse)
async def execute_code(request: ExecuteRequest):
    """Execute Python code and return output"""
    output = ""
    error = None
    
    # Capture stdout
    old_stdout = sys.stdout
    redirected = io.StringIO()
    sys.stdout = redirected
    
    try:
        # Execute the code
        exec(request.code, {"__builtins__": __builtins__})
        sys.stdout = old_stdout
        output = redirected.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        error = str(e)
    
    return ExecuteResponse(output=output, error=error)


@app.post("/api/evaluate", response_model=EvaluateResponse)
async def evaluate_answer(
    request: EvaluateRequest,
    user: str = Depends(verify_auth)
):
    """Evaluate open-ended answer using DeepSeek via Bedrock"""
    
    # Parse the bearer token for Bedrock
    if not AWS_BEARER_TOKEN:
        raise HTTPException(status_code=500, detail="AWS credentials not configured")
    
    # The token format is: "yuli-at-167004608122:base64_credential"
    # We need just the credential part for the Authorization header
    try:
        # Split by dot, take second part, decode
        parts = AWS_BEARER_TOKEN.split(".")
        if len(parts) >= 2:
            credential = parts[1]
            # Just use the full token as Bearer
            bearer_token = AWS_BEARER_TOKEN
        else:
            bearer_token = AWS_BEARER_TOKEN
    except:
        bearer_token = AWS_BEARER_TOKEN
    
    # Call Bedrock
    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id="ASdummy",  # Will be overridden by bearer token
            aws_secret_access_key="dummy",
            config=Config(signature_version="bearer")
        )
        
        # Build the prompt
        system_prompt = f"""You are a PyTorch expert tutor. Evaluate the answer to this question.

Question: {request.question}
{request.code_hint if request.code_hint else ''}

User's Answer: {request.user_answer}

Respond in JSON format:
{{"score": number 0-10, "correct": boolean, "feedback": "detailed explanation"}}"""
        
        # Call DeepSeek via Bedrock
        response = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Evaluate my answer."}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            })
        )
        
        # Parse response
        response_body = json.loads(response["body"].read())
        content = response_body["choices"][0]["message"]["content"]
        
        # Extract JSON from response
        import re
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            result = json.loads(match.group())
        else:
            result = {
                "score": 5,
                "correct": True,
                "feedback": content[:200]
            }
        
        return EvaluateResponse(**result)
        
    except Exception as e:
        return EvaluateResponse(
            score=0,
            correct=False,
            feedback=f"Evaluation failed: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pytorch_version": torch.__version__,
        "users": len(USERS)
    }


# ============== RUN ==============
if __name__ == "__main__":
    import sys
    import io
    import uvicorn
    
    print("🚀 Starting PyTorch Practice Backend")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Users: {len(USERS)}")
    print("\n   Endpoints:")
    print("   - POST /api/execute  (code execution)")
    print("   - POST /api/evaluate (AI evaluation, requires auth)")
    print("   - GET  /api/health   (health check)")
    print("\n   To test locally:")
    print('   curl -X POST http://localhost:8000/api/execute -H "Content-Type: application/json" -d \'{"code":"import torch; print(torch.randn(3,3))"}\'')
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
