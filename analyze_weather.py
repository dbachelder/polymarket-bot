#!/usr/bin/env python3
import json
from datetime import datetime

with open('data/snapshot_weather_20260217T102004Z.json') as f:
    data = json.load(f)

print(f"Snapshot: {data['generated_at']}")  
print(f"Markets: {len(data['markets'])}")
print()

for i, market in enumerate(data['markets']):
    question = market['question']
    books = market.get('books', {})
    
    # Check if we have yes/no books
    yes_book = books.get('yes')
    no_book = books.get('no') 
    
    print(f"{i+1}. {question}")
    
    yes_best_bid = None
    no_best_bid = None
    
    if yes_book and yes_book.get('bids'):
        yes_best_bid = float(yes_book['bids'][0]['price'])
        yes_size = float(yes_book['bids'][0]['size'])
        print(f"   YES bid: {yes_best_bid:.3f} (${yes_size:.0f})")
    else:
        print("   YES bid: N/A")
        
    if no_book and no_book.get('bids'):
        no_best_bid = float(no_book['bids'][0]['price'])  
        no_size = float(no_book['bids'][0]['size'])
        print(f"   NO bid:  {no_best_bid:.3f} (${no_size:.0f})")
    else:
        print("   NO bid: N/A")
    
    # Check for arbitrage opportunities    
    if yes_best_bid is not None and no_best_bid is not None:
        total_prob = yes_best_bid + no_best_bid
        arbitrage = abs(1.0 - total_prob)
        if arbitrage > 0.05:  # 5%+ arbitrage
            print(f"   >>> ARBITRAGE: {arbitrage:.3f} ({arbitrage*100:.1f}%)")
            
    print()