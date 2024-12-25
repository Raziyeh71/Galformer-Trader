# 🚀 Galformer Trader

An innovative stock trading system combining Graph Neural Networks with transformer architecture for real-time market analysis and AI-driven trading recommendations.

## 🌟 Features

- **Graph Neural Network (GNN)** for capturing market relationships
- **Real-time Analysis** of stock prices and market trends
- **AI-Powered Sentiment Analysis** of market news using LLM
- **Smart Early Stopping** to prevent overfitting
- **Advanced Technical Analysis** with dropout layers
- **Beautiful Terminal UI** for real-time trading insights

## 🔧 Technical Stack

- PyTorch for deep learning
- NetworkX for graph operations
- Rich for beautiful terminal interface
- OpenAI API for sentiment analysis
- Yahoo Finance data integration

## 📊 Model Architecture

- **GalformerModel**: Custom GNN with transformer-inspired architecture
- **Early Stopping**: Smart training optimization
- **Dropout Layers**: Preventing overfitting
- **Real-time Feature Engineering**: Dynamic market analysis

## 🚦 Getting Started

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Set up your OpenAI API key in `.env`:
```bash
OPENAI_API_KEY=your_api_key_here
```
4. Run the trader:
```bash
python stockApp.py
```

## 📈 Example Output

```
┌─────────── Stock Analysis ───────────┐
│ AAPL: Strong Buy (Confidence: 85%)   │
│ MSFT: Hold (Confidence: 72%)         │
│ GOOGL: Buy (Confidence: 78%)         │
└─────────────────────────────────────┘
```

## 🎯 Future Enhancements

- Backtesting framework
- Portfolio optimization
- Risk management system
- Advanced market indicators
- Real-time trading integration

## 📝 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions are welcome! Feel free to submit PRs or open issues.

---
Built with 💙 and Python
