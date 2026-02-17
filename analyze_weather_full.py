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
    
    # Check both bids and asks
    yes_best_bid = yes_book.get('bids', [{}])[0].get('price') if yes_book and yes_book.get('bids') else None
    yes_best_ask = yes_book.get('asks', [{}])[0].get('price') if yes_book and yes_book.get('asks') else None
    no_best_bid = no_book.get('bids', [{}])[0].get('price') if no_book and no_book.get('bids') else None  
    no_best_ask = no_book.get('asks', [{}])[0].get('price') if no_book and no_book.get('asks') else None
    
    if yes_best_bid:
        yes_bid_size = yes_book['bids'][0]['size']
        print(f"   YES bid: {yes_best_bid} (${yes_bid_size})")
    if yes_best_ask:
        yes_ask_size = yes_book['asks'][0]['size'] 
        print(f"   YES ask: {yes_best_ask} (${yes_ask_size})")
    if no_best_bid:
        no_bid_size = no_book['bids'][0]['size']
        print(f"   NO bid:  {no_best_bid} (${no_bid_size})")
    if no_best_ask:
        no_ask_size = no_book['asks'][0]['size']
        print(f"   NO ask:  {no_best_ask} (${no_ask_size})")
    
    # Only analyze if we have some liquidity
    if (yes_best_bid or yes_best_ask) and (no_best_bid or no_best_ask):
        # Convert to float for calculations
        ybid = float(yes_best_bid) if yes_best_bid else 0
        yask = float(yes_best_ask) if yes_best_ask else 1
        nbid = float(no_best_bid) if no_best_bid else 0  
        nask = float(no_best_ask) if no_best_ask else 1
        
        # Check for obvious arbitrage (sum of best bids > 1)
        if ybid + nbid > 1.0:
            arb = (ybid + nbid) - 1.0
            print(f"   >>> BID ARBITRAGE: {arb:.3f} ({arb*100:.1f}%)")
        
        # Check for obvious arbitrage (sum of best asks < 1) 
        if yask + nask < 1.0:
            arb = 1.0 - (yask + nask)
            print(f"   >>> ASK ARBITRAGE: {arb:.3f} ({arb*100:.1f}%)")
            
    print()