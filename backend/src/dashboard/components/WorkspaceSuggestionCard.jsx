// src/dashboard/components/WorkspaceSuggestionCard.jsx
import React, { useEffect, useState } from 'react';
import './WorkspaceSuggestionCard.css'; // We'll add minimal CSS inline below

export default function WorkspaceSuggestionCard({ suggestion, onAccept, onDismiss }) {
    return (
        <div className="suggestion-card">
            <h3>{suggestion.title}</h3>
            <p>{suggestion.description}</p>
            <div className="resources">
                {suggestion.resources.map((res, idx) => (
                    <span key={idx} className="resource-badge">{res.type}</span>
                ))}
            </div>
            <div className="actions">
                <button className="accept-btn" onClick={() => onAccept(suggestion.id)}>Accept</button>
                <button className="dismiss-btn" onClick={() => onDismiss(suggestion.id)}>Dismiss</button>
            </div>
        </div>
    );
}

// Minimal CSS (could be extracted to a .css file)
const style = `
.suggestion-card {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 1.5rem;
  margin: 1rem;
  color: #fff;
  box-shadow: 0 4px 30px rgba(0,0,0,0.5);
  animation: fadeIn 0.3s ease-out;
}
.suggestion-card h3 { margin-top:0; font-size:1.2rem; }
.suggestion-card .resources { margin:0.5rem 0; }
.resource-badge { background:rgba(0,0,0,0.2); padding:0.2rem 0.5rem; border-radius:4px; margin-right:0.3rem; font-size:0.9rem; }
.actions { display:flex; gap:0.5rem; }
.accept-btn, .dismiss-btn { flex:1; padding:0.5rem; border:none; border-radius:6px; cursor:pointer; }
.accept-btn { background:#4caf50; color:#fff; }
.dismiss-btn { background:#f44336; color:#fff; }
@keyframes fadeIn { from { opacity:0; transform:scale(0.95); } to { opacity:1; transform:scale(1); } }
`;

// Inject style into document head (for simplicity in this demo)
if (typeof document !== 'undefined') {
    const styleTag = document.createElement('style');
    styleTag.innerHTML = style;
    document.head.appendChild(styleTag);
}
