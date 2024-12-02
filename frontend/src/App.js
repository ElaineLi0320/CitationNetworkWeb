import React, { useState, useEffect } from "react";
import axios from "axios";
import { Network } from "vis-network";
import "./App.css";

const App = () => {
  const [keyword, setKeyword] = useState("");
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [path, setPath] = useState([]);
  const [articleDetails, setArticleDetails] = useState(null);

  const handleFilter = async () => {
    if (!keyword) {
      alert("Please enter a keyword!");
      return;
    }

    try {
      const response = await axios.post("http://127.0.0.1:5000/api/filter", {
        keyword,
      });
      if (response.data.success) {
        setNodes(
          response.data.nodes.map((node) => ({
            id: node.Id,
            label: node.Label,
            title: `Title: ${node.Label}\nAuthors: ${node.Authors}\nYear: ${node.Year}`,
          }))
        );
        setEdges([]); // Reset edges if only filtering nodes
      } else {
        alert(response.data.message);
      }
    } catch (error) {
      console.error("Error filtering nodes:", error);
    }
  };

  const handleFindPath = async () => {
    if (!selectedNode) {
      alert("Please select a start node!");
      return;
    }
    try {
      const response = await axios.post("http://127.0.0.1:5000/api/path", {
        start_point: selectedNode.id,
        keyword,
      });
      if (response.data.success) {
        const newEdges = response.data.path
          .map((_, i, arr) => {
            if (i < arr.length - 1) {
              return { from: arr[i].Id, to: arr[i + 1].Id };
            }
            return null;
          })
          .filter(Boolean);

        setPath(response.data.path);
        setEdges(newEdges);
      } else {
        alert(response.data.message);
      }
    } catch (error) {
      console.error("Error finding path:", error);
    }
  };

  useEffect(() => {
    if (nodes.length) {
      const container = document.getElementById("network");
      const data = {
        nodes,
        edges,
      };
      const options = {
        physics: false,
        nodes: {
          shape: "dot",
          size: 10,
          color: {
            background: "#FFFFFF",
            border: "#9C27B0",
          },
          borderWidth: 2,
          font: {
            color: "#D8B4FF",
            size: 14,
          },
        },
        edges: {
          arrows: {
            to: { enabled: true, scaleFactor: 1 },
          },
          color: {
            color: "#9C27B0",
          },
          width: 2,
        },
        interaction: {
          hover: true,
        },
      };
      const network = new Network(container, data, options);

      network.on("selectNode", (params) => {
        if (params.nodes.length) {
          const nodeId = params.nodes[0];
          const selectedNodeDetails = nodes.find((node) => node.id === nodeId);
          setSelectedNode(selectedNodeDetails);
          setArticleDetails({
            id: selectedNodeDetails.id,
            label: selectedNodeDetails.label,
            title: selectedNodeDetails.title,
          });
        }
      });
    }
  }, [nodes, edges]);

  return (
    <div className="app-container">
      <h1 className="app-title">Citation Recommendation System</h1>
      <div className="search-container">
        <input
          type="text"
          className="search-input"
          placeholder="Enter keyword"
          value={keyword}
          onChange={(e) => setKeyword(e.target.value)}
        />
        <button className="search-button" onClick={handleFilter}>
          Search
        </button>
      </div>
      {selectedNode && (
        <div className="selected-node-details">
          <h3>Selected Article Detail</h3>
          <p>Id: {selectedNode.id}</p>
          <p>Title: {selectedNode.label}</p>
          <p>Authors: {selectedNode.title.split("\n")[1]}</p>
        </div>
      )}
      <div id="network" className="network-container"></div>
      <div className="actions-container">
        <button
          className="path-button"
          onClick={handleFindPath}
          disabled={!selectedNode}
        >
          Find Path
        </button>
      </div>
      {path.length > 0 && (
        <div className="result-container">
          <h3 className="result-title">Path Details:</h3>
          <p>{path.map((node) => node.Id).join(" -> ")}</p>
          <ul className="path-list">
            {path.map((node, index) => (
              <li key={index}>
                <p>Id: {node.Id}</p>
                <p>Title: {node.Label}</p>
                <p>Authors: {node.Authors}</p>
                <p>Year: {node.Year}</p>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default App;
