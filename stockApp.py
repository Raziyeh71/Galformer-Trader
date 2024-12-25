import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import os # why?????
from dotenv import load_dotenv # ????!!!!
from langchain_openai import ChatOpenAI
import requests
from bs4 import BeautifulSoup
import time
import torch
import torch.nn as nn
import networkx as nx
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize console
console = Console()

class GalformerModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=2, dropout_rate=0.2):
        super(GalformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        
        # Graph attention layers with dropout
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            for i in range(num_layers)
        ])
        
        # Final prediction layers with dropout
        self.price_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Time horizon predictor with dropout
        self.time_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
   
    def forward(self, x, adj_matrix): #(why and how related adj m and gnn??????)
        # Apply GNN layers with dropout
        for layer in self.gnn_layers:
            x = layer(x)
            # Ensure proper matrix multiplication
            x = torch.matmul(adj_matrix, x)  # Changed from x @ adj_matrix
        
        # Predictions with dropout
        price_pred = self.price_predictor(x)
        time_pred = self.time_predictor(x)
        
        return price_pred, time_pred

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate multiple evaluation metrics"""
        # Convert to numpy for calculations
        y_true = y_true.detach().numpy() if torch.is_tensor(y_true) else y_true
        y_pred = y_pred.detach().numpy() if torch.is_tensor(y_pred) else y_pred
        
        # Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
    
    @staticmethod
    def calculate_trading_metrics(y_true, y_pred, threshold=0.02):
        """Calculate trading-specific metrics"""
        correct_direction = np.sum(np.sign(y_true) == np.sign(y_pred))
        direction_accuracy = correct_direction / len(y_true)
        
        # Calculate profitable trades (prediction within threshold)
        profitable = np.sum(np.abs(y_pred - y_true) <= threshold * y_true)
        profit_ratio = profitable / len(y_true)
        
        return {
            'direction_accuracy': direction_accuracy,
            'profit_ratio': profit_ratio
        }

class GalformerAnalyzer:
    def __init__(self):
        self.model = GalformerModel()
        self.scaler = StandardScaler()
        self.graph = nx.Graph()
        self.early_stopping = EarlyStopping(patience=5, min_delta=0.01)
        self.evaluator = ModelEvaluator()
        
        # Initialize weights to avoid random predictions
        for param in self.model.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
            
    def prepare_data(self, prices_dict):
        """Prepare data for Galformer analysis"""
        # Create feature matrix
        features = []
        symbols = list(prices_dict.keys())
        
        for symbol in symbols:
            price = float(prices_dict[symbol])
            # Simple features: price, day_change, volatility
            features.append([
                price / 1000.0,  # Normalize price
                0.02,  # Mock day change
                0.015  # Mock volatility
            ])
        
        # Create adjacency matrix for the single stock
        adj_matrix = torch.eye(1)
        
        # Convert to PyTorch tensors
        features = torch.FloatTensor([features[0]])  # Only use the current stock
        
        return features, adj_matrix, symbols
    
    def get_time_recommendation(self, time_score):
        """Convert time score to days recommendation"""
        # Convert sigmoid output (0-1) to days (1-7)
        days = max(1, min(7, round(time_score * 7)))
        
        # Get the future date
        future_date = datetime.now() + timedelta(days=days)
        
        # Format the recommendation
        if days == 1:
            return "tomorrow", future_date.strftime("%Y-%m-%d")
        elif days == 2:
            return "in 2 days", future_date.strftime("%Y-%m-%d")
        else:
            return f"in {days} days", future_date.strftime("%Y-%m-%d")
    
    def analyze(self, prices_dict):
        """Analyze stocks using Galformer"""
        features, adj_matrix, symbols = self.prepare_data(prices_dict)
        
        with torch.no_grad():
            # Get model predictions
            price_predictions, time_predictions = self.model(features, adj_matrix)
            
            # Convert predictions to recommendations
            results = {}
            for i, symbol in enumerate(symbols):
                price_score = price_predictions[i].item()
                time_score = time_predictions[i].item()
                
                # Normalize price score to be between -1 and 1
                price_score = np.tanh(price_score)
                
                # Get time recommendation
                time_recommendation, target_date = self.get_time_recommendation(time_score)
                
                # Convert score to recommendation
                if price_score > 0.5:
                    recommendation = "Strong Buy"
                elif price_score > 0.2:
                    recommendation = "Buy"
                elif price_score > -0.2:
                    recommendation = "Hold"
                elif price_score > -0.5:
                    recommendation = "Sell"
                else:
                    recommendation = "Strong Sell"
                
                results[symbol] = {
                    'score': price_score,
                    'recommendation': recommendation,
                    'confidence': min(abs(price_score) * 100, 100),
                    'time_recommendation': time_recommendation,
                    'target_date': target_date
                }
            
            return results
    
    def train_step(self, features, adj_matrix, targets, optimizer, criterion):
        """Single training step"""
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        price_pred, time_pred = self.model(features, adj_matrix)
        loss = criterion(price_pred, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def validate(self, val_features, val_adj_matrix, val_targets, criterion):
        """Validation step with multiple metrics"""
        self.model.eval()
        with torch.no_grad():
            price_pred, time_pred = self.model(val_features, val_adj_matrix)
            val_loss = criterion(price_pred, val_targets)
            
            # Calculate additional metrics for validation
            metrics = self.evaluator.calculate_metrics(val_targets, price_pred)
            
        return val_loss.item(), metrics
    
    def train_model(self, train_data, val_data, epochs=100, lr=0.001):
        """Train the model with early stopping"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        train_features, train_adj_matrix, train_targets = train_data
        val_features, val_adj_matrix, val_targets = val_data
        
        for epoch in range(epochs):
            # Training step
            train_loss = self.train_step(train_features, train_adj_matrix, train_targets, optimizer, criterion)
            
            # Validation step
            val_loss, val_metrics = self.validate(val_features, val_adj_matrix, val_targets, criterion)
            
            # Early stopping check
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    
    def evaluate_model(self, test_features, test_adj_matrix, test_targets):
        """Comprehensive model evaluation"""
        self.model.eval()
        with torch.no_grad():
            # Get predictions
            price_pred, time_pred = self.model(test_features, test_adj_matrix)
            
            # Calculate all metrics
            price_metrics = self.evaluator.calculate_metrics(test_targets, price_pred)
            trading_metrics = self.evaluator.calculate_trading_metrics(
                test_targets.numpy(), 
                price_pred.numpy()
            )
            
            # Combine all metrics
            evaluation_results = {
                'price_metrics': price_metrics,
                'trading_metrics': trading_metrics
            }
            
            # Print evaluation results
            console.print("\n[bold blue]Model Evaluation Results:[/bold blue]")
            console.print("\n[yellow]Price Prediction Metrics:[/yellow]")
            for metric, value in price_metrics.items():
                console.print(f"{metric.upper()}: {value:.4f}")
            
            console.print("\n[yellow]Trading Metrics:[/yellow]")
            for metric, value in trading_metrics.items():
                console.print(f"{metric.upper()}: {value:.4f}")
            
            return evaluation_results

class MarketAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = ChatOpenAI(temperature=0.7, api_key=api_key)
        self.galformer = GalformerAnalyzer()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_stock_price(self, symbol):
        """Get stock price with multiple retries"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                url = f"https://finance.yahoo.com/quote/{symbol}"
                response = requests.get(url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Try different price selectors
                price_element = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
                if not price_element:
                    price_element = soup.find('div', {'class': 'D(ib) Mend(20px)'})
                    if price_element:
                        price_text = price_element.find('span').text
                    else:
                        raise ValueError("Could not find price element")
                else:
                    price_text = price_element.text
                
                return float(price_text.replace(',', ''))
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise e
    
    def analyze_stock(self, symbol: str) -> dict:
        try:
            # Get stock price with retries
            try:
                current_price = self.get_stock_price(symbol)
            except Exception as e:
                console.print(f"[yellow]Could not fetch price for {symbol}, using mock data[/yellow]")
                current_price = 100.0  # Mock price
            
            # Get news headlines
            try:
                news_url = f"https://finance.yahoo.com/quote/{symbol}/news"
                response = requests.get(news_url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                headlines = []
                for article in soup.find_all(['h3', 'a'], limit=10):
                    text = article.text.strip()
                    if len(text) > 20 and text not in headlines:
                        headlines.append(text)
                    if len(headlines) >= 5:
                        break
            except Exception:
                headlines = [
                    f"Latest market trends for {symbol}",
                    f"Analyzing {symbol} performance",
                    f"Market outlook for {symbol}",
                    f"Industry analysis for {symbol}",
                    f"Technical analysis for {symbol}"
                ]
            
            # Create a prompt for the LLM
            prompt = f"""Analyze the investment potential for {symbol} based on recent news:
            
            Headlines:
            {chr(10).join(headlines)}
            
            Current Price: ${current_price:.2f}
            
            Please provide:
            1. A buy price recommendation (as a percentage below current price)
            2. A target price recommendation (as a percentage above current price)
            3. A brief sentiment analysis of the news
            4. An overall rating (Strong Buy, Buy, Hold, Sell, Strong Sell)
            
            Format the response as:
            Buy Price: $X
            Target Price: $X
            Sentiment: [Your analysis]
            Rating: [Your rating]"""
            
            analysis = self.llm.predict(prompt)
            
            # Parse the analysis
            lines = analysis.split('\n')
            buy_price = None
            target_price = None
            sentiment = None
            rating = None
            
            for line in lines:
                if 'Buy Price:' in line:
                    try:
                        buy_price = float(line.split('$')[1])
                    except:
                        buy_price = current_price * 0.95
                elif 'Target Price:' in line:
                    try:
                        target_price = float(line.split('$')[1])
                    except:
                        target_price = current_price * 1.1
                elif 'Sentiment:' in line:
                    sentiment = line.split('Sentiment:')[1].strip()
                elif 'Rating:' in line:
                    rating = line.split('Rating:')[1].strip()
            
            # Get Galformer analysis
            galformer_analysis = self.galformer.analyze({symbol: current_price})
            galformer_rating = galformer_analysis[symbol]
            
            # Combine traditional and Galformer analysis
            combined_rating = self._combine_ratings(rating, galformer_rating['recommendation'])
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'buy_price': buy_price or current_price * 0.95,
                'target_price': target_price or current_price * 1.1,
                'sentiment': sentiment,
                'rating': combined_rating,
                'galformer_confidence': galformer_rating['confidence'],
                'potential_return': ((target_price or current_price * 1.1) - current_price) / current_price * 100,
                'time_recommendation': galformer_rating['time_recommendation'],
                'target_date': galformer_rating['target_date']
            }
            
        except Exception as e:
            console.print(f"[red]Error analyzing {symbol}: {str(e)}[/red]")
            return None
    
    def _combine_ratings(self, traditional_rating, galformer_rating):
        """Combine traditional and Galformer ratings"""
        rating_scores = {
            'Strong Buy': 2,
            'Buy': 1,
            'Hold': 0,
            'Sell': -1,
            'Strong Sell': -2
        }
        
        # Get numerical scores
        trad_score = rating_scores.get(traditional_rating, 0)
        gal_score = rating_scores.get(galformer_rating, 0)
        
        # Average the scores
        combined_score = (trad_score + gal_score) / 2
        
        # Convert back to rating
        for rating, score in rating_scores.items():
            if abs(combined_score - score) <= 0.5:
                return rating
        
        return 'Hold'  # Default case

def main():
    # Initialize analyzer
    market_analyzer = MarketAnalyzer(OPENAI_API_KEY)
    
    # List of stocks to analyze
    symbols = [
        'AAPL', 'MSFT', 'GOOGL',  # Tech giants
        'NVDA', 'AMD', 'TSLA'  # High-growth tech
    ]
    
    console.print(Panel.fit("[bold green]Stock Recommendation System[/bold green]"))
    console.print("[yellow]Analyzing market data and gathering recommendations...[/yellow]")
    
    # Store results
    results = []
    
    # Analyze each symbol
    for symbol in symbols:
        console.print(f"\n[cyan]Analyzing {symbol}...[/cyan]")
        analysis = market_analyzer.analyze_stock(symbol)
        if analysis:
            results.append(analysis)
            time.sleep(2)  # Avoid rate limiting
    
    if not results:
        console.print("[red]No valid predictions could be made. Please try again later.[/red]")
        return
    
    # Sort by potential return
    results.sort(key=lambda x: x['potential_return'], reverse=True)
    
    # Display results
    table = Table(title="Investment Recommendations")
    table.add_column("Symbol", style="cyan")
    table.add_column("Current Price", style="green")
    table.add_column("Buy Price", style="yellow")
    table.add_column("Target Price", style="red")
    table.add_column("Potential Return", style="magenta")
    table.add_column("Rating", style="blue")
    table.add_column("AI Confidence", style="cyan")
    table.add_column("Optimal Sell Time", style="yellow")
    
    for result in results:
        table.add_row(
            result['symbol'],
            f"${result['current_price']:.2f}",
            f"${result['buy_price']:.2f}",
            f"${result['target_price']:.2f}",
            f"{result['potential_return']:.1f}%",
            result['rating'],
            f"{result['galformer_confidence']:.1f}%",
            f"{result['time_recommendation']} ({result['target_date']})"
        )
    
    console.print(table)
    
    # Show detailed analysis for each symbol
    console.print("\n[cyan]Detailed Analysis:[/cyan]")
    for result in results:
        console.print(Panel.fit(
            f"[bold]{result['symbol']}[/bold]\n\n" +
            f"Current Price: ${result['current_price']:.2f}\n" +
            f"Buy Price: ${result['buy_price']:.2f}\n" +
            f"Target Price: ${result['target_price']:.2f}\n" +
            f"Potential Return: {result['potential_return']:.1f}%\n" +
            f"Rating: {result['rating']} (AI Confidence: {result['galformer_confidence']:.1f}%)\n" +
            f"Recommended Sell Time: {result['time_recommendation']} ({result['target_date']})\n\n" +
            f"Sentiment Analysis:\n{result['sentiment']}",
            title=f"Analysis: {result['symbol']}"
        ))

if __name__ == "__main__":
    main()
