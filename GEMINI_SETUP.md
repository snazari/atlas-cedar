# Google Gemini AI Summary Setup

## 📋 Prerequisites

1. **Install Google Generative AI library**:
   ```bash
   pip install google-generativeai
   ```

## 🔑 Get Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Get API Key" or "Create API Key"
3. Copy your API key

## ⚙️ Set Up Environment Variable

### Option 1: Temporary (Current Session Only)
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Option 2: Permanent (Add to your shell profile)

**For Bash** (`~/.bashrc` or `~/.bash_profile`):
```bash
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**For Zsh** (`~/.zshrc`):
```bash
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### Option 3: Set in your script
Add to `get_latest_spike.sh` before running streamlit:
```bash
export GEMINI_API_KEY="your-api-key-here"
/opt/anaconda3/envs/gt/bin/streamlit run streamlit_results_viewer.py
```

## 🚀 Usage

1. Run your streamlit dashboard:
   ```bash
   streamlit run streamlit_results_viewer.py
   ```

2. Navigate to the **"📈 Detailed Charts"** tab

3. Scroll down to any asset section

4. Click the **"🤖 Generate AI Summary"** button

5. Wait a few seconds for the AI-powered analysis!

## 💰 Pricing

- **Gemini Pro** (used in this implementation):
  - Free tier: 60 requests per minute
  - Very affordable for personal use
  - [Pricing details](https://ai.google.dev/pricing)

## 🔒 Security Note

- Never commit your API key to version control
- Use environment variables or secret management
- The `.gitignore` should exclude any files with API keys

## 🐛 Troubleshooting

**Error: "Gemini API key not set"**
- Make sure you've exported the `GEMINI_API_KEY` environment variable
- Restart your terminal/shell after setting it

**Error: "Google Gemini library not installed"**
- Run: `pip install google-generativeai`

**API Rate Limit Errors**
- Free tier has limits (60 requests/minute)
- Wait a minute and try again
- Consider upgrading if you need higher limits
