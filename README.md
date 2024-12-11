# Paper Learning Path Recommendation System

This project is a web-based application that provides interactive tools for exploring and analyzing citation networks. It leverages Flask for the backend, React for the frontend, and libraries such as NetworkX and Pandas for graph analysis and data manipulation.

## Features
- **Interactive Citation Network**: Visualize and interact with citation networks.
- **Search Functionality**: Search for specific nodes and view their details.
- **Shortest Path Finder**: Compute the shortest path between nodes in the network.
- **Dynamic Frontend**: Built with React for an engaging user experience.
- **REST API**: Backend endpoints to handle requests and process data.

## 1 Data Description

### Dataset Statistics
- Source file:https://www.aminer.cn/citation V1
- Original sample:
  First 100,000 papers from the dataset
- After filtered: 5248 papers and 2601 edges

### Output Files
- Edge data: `citation_network_edges2`
- Node data: `citation_network_nodes2`

## 2 Data Handling

### 1. Data Processing (dataProcess.py)
Handles data preprocessing and filtering to generate standardized DataFrames.

**Features:**
- Data cleaning and standardization
- DataFrame construction
- Data filtering and preprocessing

### 2. Gephi Export (networkGephi.py)
Generates network files compatible with Gephi software.

**Outputs:**
- Node file (nodes.csv)
- Edge file (edges.csv)

**Advantages:**
- Supports large-scale datasets
- Compatible with professional network analysis tool Gephi

### Usage Workflow
1. Run dataProcess.py to process raw data
2. Use networkGephi.py to generate files for Gephi and further application usage

### Requirements

```python
networkx>=2.6
pandas>=1.3.0
numpy>=1.20.0
```


## 3 Website Installation

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
