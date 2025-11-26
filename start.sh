#!/bin/bash

# Credit Card Fraud Detection - Startup Script

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Credit Card Fraud Detection System                   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"

case "$1" in
    backend)
        echo -e "${GREEN}Starting Backend API...${NC}"
        cd backend
        if [ ! -d "venv" ]; then
            echo -e "${YELLOW}Creating virtual environment...${NC}"
            python3 -m venv venv
        fi
        source venv/bin/activate
        pip install -r requirements.txt -q
        python app.py
        ;;
    
    frontend)
        echo -e "${GREEN}Starting Frontend...${NC}"
        cd frontend
        npm install
        npx ng serve --open
        ;;
    
    train)
        echo -e "${GREEN}Training Model...${NC}"
        cd backend
        if [ ! -d "venv" ]; then
            python3 -m venv venv
        fi
        source venv/bin/activate
        pip install -r requirements.txt -q
        
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Usage: ./start.sh train <path_to_creditcard.csv>${NC}"
            echo -e "${YELLOW}Download dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud${NC}"
            exit 1
        fi
        
        python model.py "$2"
        ;;
    
    install)
        echo -e "${GREEN}Installing all dependencies...${NC}"
        
        echo -e "${BLUE}Installing backend dependencies...${NC}"
        cd backend
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
        cd ..
        
        echo -e "${BLUE}Installing frontend dependencies...${NC}"
        cd frontend
        npm install
        cd ..
        
        echo -e "${GREEN}✓ All dependencies installed!${NC}"
        ;;
    
    *)
        echo -e "${YELLOW}Usage: ./start.sh {backend|frontend|train|install}${NC}"
        echo ""
        echo "Commands:"
        echo "  backend   - Start the Flask API server (port 5000)"
        echo "  frontend  - Start the Angular dev server (port 4200)"
        echo "  train     - Train the PyTorch model"
        echo "  install   - Install all dependencies"
        echo ""
        echo "Quick Start:"
        echo "  1. ./start.sh install"
        echo "  2. ./start.sh train path/to/creditcard.csv"
        echo "  3. ./start.sh backend  (in terminal 1)"
        echo "  4. ./start.sh frontend (in terminal 2)"
        ;;
esac
