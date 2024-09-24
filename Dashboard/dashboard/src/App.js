import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Home from './components/Home/Home';
import FileClassifier from './components/FileClassifier/FileClassifier';
import './App.css'

function App() {
  return (
    <div>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/classify" element={<FileClassifier />} />
      </Routes>
    </div>
  );
}

export default App;
