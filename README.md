# Citation Network Web

This project is a web-based application that provides interactive tools for exploring and analyzing citation networks. It leverages Flask for the backend, React for the frontend, and libraries such as NetworkX and Pandas for graph analysis and data manipulation.

## Features
- **Interactive Citation Network**: Visualize and interact with citation networks.
- **Search Functionality**: Search for specific nodes and view their details.
- **Shortest Path Finder**: Compute the shortest path between nodes in the network.
- **Dynamic Frontend**: Built with React for an engaging user experience.
- **REST API**: Backend endpoints to handle requests and process data.

## Installation

### Prerequisites
- Python 3.7+
- Node.js 16+

### Clone the Repository
```bash
git clone https://github.com/ElaineLi0320/CitationNetworkWeb.git
cd CitationNetworkWeb
```

### Backend Setup
1. Navigate to the backend folder (if applicable) or stay in the root directory.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Flask backend:
   ```bash
   python app.py
   ```

### Frontend Setup
1. Navigate to the frontend directory (e.g., `client`).
2. Install Node.js dependencies:
   ```bash
   npm install
   ```
3. Start the React development server:
   ```bash
   npm start
   ```

### Access the Application
Once both the backend and frontend are running, open your browser and navigate to:
```
http://localhost:3000
```

## Project Structure
- `app.py`: Main backend logic for serving data and handling requests.
- `client/`: Frontend React application.
- `requirements.txt`: Python dependencies for the backend.
- `App.css` and `App.js`: Frontend styles and main logic.
