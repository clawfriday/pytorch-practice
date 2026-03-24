# PyTorch Practice - Interactive Learning Platform

An interactive web application to practice PyTorch functions with AI-powered evaluation.

## Features

- 📝 **Question Bank**: Questions parsed from PyTorch documentation
- 🔒 **MCQ & Open-ended**: Multiple choice and open-ended questions
- 🤖 **AI Evaluation**: DeepSeek API integration for open-ended answer review
- 💻 **Interactive Terminal**: Run PyTorch code directly in the browser using Pyodide
- 📊 **Progress Tracking**: Track your learning progress

## Tech Stack

- **Frontend**: Vue.js 3 + Tailwind CSS
- **Code Execution**: Pyodide (Python in WebAssembly)
- **AI Evaluation**: DeepSeek API

## Setup

1. **Clone the repository**
```bash
git clone https://github.com/clawfriday/pytorch-practice.git
cd pytorch-practice
```

2. **Open in browser**
Simply open `index.html` in your browser, or serve via a local server:

```bash
# Using Python
python -m http.server 8000

# Then open http://localhost:8000
```

## Usage

1. Enter your DeepSeek API key (required for open-ended question evaluation)
2. Select a topic from the dropdown
3. Answer questions:
   - **MCQ**: Click on the correct option
   - **Open-ended**: Type your answer and click "Submit for AI Review"
4. Practice code in the Python terminal

## API Key Setup

To evaluate open-ended answers, you need a DeepSeek API key:

1. Get your API key from [DeepSeek](https://platform.deepseek.com/)
2. Enter it in the application (stored locally, never sent to external servers except DeepSeek)

## File Structure

```
pytorch-practice/
├── index.html          # Main application
├── data/
│   └── questions.json  # Question bank
├── .github/
│   └── workflows/
│       └── deploy.yml  # GitHub Pages deployment
└── README.md
```

## License

MIT