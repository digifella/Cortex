#!/bin/bash
# ## File: setup_mistral_small.sh
# Purpose: Setup script to install and configure Mistral Small 3.2 for proposal generation
# Date: 2025-07-24

echo "🚀 Setting up Mistral Small 3.2 for Cortex Suite Proposal Generation"
echo "=================================================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed. Please install it first:"
    echo "   curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

echo "✅ Ollama is installed"

# Check if Ollama service is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "🔄 Starting Ollama service..."
    ollama serve &
    sleep 5
fi

echo "✅ Ollama service is running"

# Pull Mistral Small 3.2 model
echo "📥 Downloading Mistral Small 3.2 model (this may take a while...)"
echo "   Model size: ~24GB for the full model"
echo "   Note: This requires significant disk space and RAM (55GB+ for full precision)"

# Check available disk space
available_space=$(df . | awk 'NR==2 {print $4}')
required_space=$((30 * 1024 * 1024))  # 30GB in KB

if [ "$available_space" -lt "$required_space" ]; then
    echo "⚠️  Warning: You may not have enough disk space (need ~30GB)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Pull the model
if ollama pull mistral-small:3.2; then
    echo "✅ Mistral Small 3.2 downloaded successfully"
else
    echo "❌ Failed to download Mistral Small 3.2"
    echo "   You may want to try a quantized version instead:"
    echo "   ollama pull mistral-small"
    exit 1
fi

# Test the model
echo "🧪 Testing Mistral Small 3.2..."
test_response=$(ollama run mistral-small:3.2 "Write a one-sentence professional proposal introduction." --timeout 30)

if [ $? -eq 0 ]; then
    echo "✅ Model test successful!"
    echo "📝 Test response: $test_response"
else
    echo "❌ Model test failed"
    exit 1
fi

echo ""
echo "🎉 Setup Complete!"
echo "=================================================================="
echo "Mistral Small 3.2 is now ready for use in your Cortex Suite."
echo ""
echo "To enable it:"
echo "1. Edit your .env file and change LLM_PROVIDER to 'ollama'"
echo "2. Restart your Cortex Suite application"
echo ""
echo "Benefits for proposal generation:"
echo "• 84% better instruction following"
echo "• 50% reduction in repetitive outputs"
echo "• Improved consistency and reliability"
echo "• Faster generation (150 tokens/s)"
echo ""
echo "Memory requirements:"
echo "• ~55GB RAM for full precision (bf16/fp16)"
echo "• Consider using quantized versions for lower memory usage"