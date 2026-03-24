#!/usr/bin/env python3
"""
Parse day0.ipynb and generate questions.json
"""
import json
import re

# Read the notebook
with open('/workspace/doc/pytorch/day0.ipynb', 'r') as f:
    notebook = json.load(f)

questions = []
topic_id = 0

def extract_concepts(code):
    """Extract key PyTorch concepts from code"""
    concepts = []
    patterns = {
        r'torch\.\w+': 'torch function',
        r'nn\.\w+': 'nn.Module',
        r'F\.\w+': 'torch.nn.functional',
        r'\.(\w+)\(': 'method call',
    }
    
    for pattern, concept in patterns.items():
        if re.search(pattern, code):
            concepts.append(concept)
    
    return list(set(concepts))

for cell in notebook['cells']:
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        
        # Check if it's a section header (## X. Topic Name)
        match = re.match(r'##\s+(\d+)\.?\s*(.+?)(?:\n|$)', source)
        if match:
            topic_id = int(match.group(1))
            topic_name = match.group(2).strip()
            current_topic = topic_name
            continue
        
        # Check for subsection (###)
        match = re.match(r'###\s+(.+?)(?:\n|$)', source)
        if match:
            current_subtopic = match.group(1).strip()
            continue
    
    elif cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Skip if it's just imports or short snippets
        if len(source) < 50:
            continue
        
        # Extract comments and code for question generation
        comments = re.findall(r'#\s*(.+?)(?:\n|$)', source)
        
        if comments:
            # Generate question from this code cell
            question = {
                "id": len(questions) + 1,
                "topic": current_topic if 'current_topic' in locals() else "General",
                "type": "open-ended",
                "question": f"Explain what the following PyTorch code does and write equivalent code to achieve the same result:",
                "code": source.strip(),
                "hints": comments[:3] if comments else [],
                "expected_concepts": extract_concepts(source)
            }
            questions.append(question)

def extract_concepts(code):
    """Extract key PyTorch concepts from code"""
    concepts = []
    patterns = {
        r'torch\.\w+': 'torch function',
        r'nn\.\w+': 'nn.Module',
        r'F\.\w+': 'torch.nn.functional',
        r'\.(\w+)\(': 'method call',
    }
    
    for pattern, concept in patterns.items():
        if re.search(pattern, code):
            concepts.append(concept)
    
    return list(set(concepts))

# Generate questions based on topics (more comprehensive)
question_bank = [
    # Tensor Creation
    {
        "id": 1,
        "topic": "Tensor Creation",
        "type": "mcq",
        "question": "Which PyTorch function creates a tensor with random values from a standard normal distribution?",
        "options": ["torch.rand()", "torch.randn()", "torch.randperm()", "torch.random()"],
        "correct": "B",
        "explanation": "torch.randn() creates tensors with values sampled from a standard normal distribution N(0,1)."
    },
    {
        "id": 2,
        "topic": "Tensor Creation",
        "type": "open-ended",
        "question": "Create a 3x3 identity matrix using PyTorch. Explain the function you used.",
        "code_hint": "torch.eye(3)",
        "expected_output": "A 3x3 identity matrix with 1s on diagonal"
    },
    {
        "id": 3,
        "topic": "Tensor Creation",
        "type": "mcq",
        "question": "What is the difference between torch.tensor() and torch.from_numpy()?",
        "options": [
            "No difference",
            "tensor() copies data, from_numpy() shares memory",
            "from_numpy() copies data, tensor() shares memory",
            "tensor() only works with lists"
        ],
        "correct": "B",
        "explanation": "torch.from_numpy() shares memory with the NumPy array, while torch.tensor() creates a copy by default."
    },
    
    # Tensor Operations
    {
        "id": 4,
        "topic": "Tensor Operations",
        "type": "open-ended",
        "question": "Write code to reshape a tensor from shape (2, 6) to (3, 4). What is the difference between view() and reshape()?",
        "code_hint": "t.view(3, 4) or t.reshape(3, 4)"
    },
    {
        "id": 5,
        "topic": "Tensor Operations",
        "type": "mcq",
        "question": "What does .squeeze() do in PyTorch?",
        "options": [
            "Adds a dimension of size 1",
            "Removes all dimensions of size 1",
            "Transposes the tensor",
            "Flattens the tensor"
        ],
        "correct": "B",
        "explanation": ".squeeze() removes all dimensions that have size 1. Use .unsqueeze() to add a dimension."
    },
    {
        "id": 6,
        "topic": "Tensor Operations",
        "type": "mcq",
        "question": "Which operation is used for matrix multiplication in PyTorch?",
        "options": ["torch.mul()", "torch.matmul()", "torch.dot()", "torch.cross()"],
        "correct": "B",
        "explanation": "torch.matmul() (or @ operator) performs matrix multiplication. torch.mul() is element-wise."
    },
    
    # Autograd
    {
        "id": 7,
        "topic": "Autograd",
        "type": "open-ended",
        "question": "Create a tensor with requires_grad=True and compute the gradient of y = x^2 at x=3.",
        "code_hint": "x = torch.tensor([3.0], requires_grad=True); y = x**2; y.backward()",
        "expected_output": "The gradient should be 2*3 = 6"
    },
    {
        "id": 8,
        "topic": "Autograd",
        "type": "mcq",
        "question": "What does optimizer.zero_grad() do?",
        "options": [
            "Sets all gradients to None",
            "Sets all gradients to zero before backpropagation",
            "Deletes the computational graph",
            "Initializes weights to zero"
        ],
        "correct": "B",
        "explanation": "zero_grad() clears the gradients of all parameters. Called before each training step."
    },
    {
        "id": 9,
        "topic": "Autograd",
        "type": "mcq",
        "question": "When should you use torch.no_grad()?",
        "options": [
            "During training",
            "During inference/evaluation",
            "When computing gradients",
            "Always"
        ],
        "correct": "B",
        "explanation": "no_grad() disables gradient computation to save memory and speed up inference."
    },
    
    # nn.Module
    {
        "id": 10,
        "topic": "nn.Module",
        "type": "open-ended",
        "question": "Create a simple neural network with one hidden layer using nn.Module.",
        "code_hint": """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)"""
    },
    {
        "id": 11,
        "topic": "nn.Module",
        "type": "mcq",
        "question": "Which method returns all trainable parameters in a nn.Module?",
        "options": ["model.parameters()", "model.state_dict()", "model.modules()", "model.children()"],
        "correct": "A",
        "explanation": "model.parameters() returns an iterator over all trainable parameters."
    },
    {
        "id": 12,
        "topic": "nn.Module",
        "type": "mcq",
        "question": "What does model.train() do?",
        "options": [
            "Starts training mode",
            "Sets dropout and batch norm to training mode",
            "Loads training data",
            "Enables gradient computation"
        ],
        "correct": "B",
        "explanation": "train() sets dropout and batch norm layers to training mode. Use eval() for inference."
    },
    
    # Loss Functions
    {
        "id": 13,
        "topic": "Loss Functions",
        "type": "mcq",
        "question": "Which loss function is used for multi-class classification?",
        "options": ["MSELoss", "BCELoss", "CrossEntropyLoss", "L1Loss"],
        "correct": "C",
        "explanation": "CrossEntropyLoss is used for multi-class classification problems."
    },
    {
        "id": 14,
        "topic": "Loss Functions",
        "type": "open-ended",
        "question": "Calculate the CrossEntropyLoss between predicted logits [2.0, 1.0, 0.1] and ground truth class 0.",
        "code_hint": "F.cross_entropy(torch.tensor([[2.0, 1.0, 0.1]]), torch.tensor([0]))"
    },
    
    # Optimizers
    {
        "id": 15,
        "topic": "Optimizers",
        "type": "mcq",
        "question": "What is the default learning rate for Adam optimizer?",
        "options": ["0.001", "0.01", "0.0001", "There is no default"],
        "correct": "D",
        "explanation": "You must explicitly specify the learning rate - there's no default."
    },
    {
        "id": 16,
        "topic": "Optimizers",
        "type": "open-ended",
        "question": "Create an Adam optimizer with learning rate 0.001 for a model with parameters.",
        "code_hint": "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)"
    },
]

# Save questions
with open('/workspace/repos/pytorch-practice/data/questions.json', 'w') as f:
    json.dump(question_bank, f, indent=2, ensure_ascii=False)

print(f"Generated {len(question_bank)} questions")
print("Saved to: /workspace/repos/pytorch-practice/data/questions.json")