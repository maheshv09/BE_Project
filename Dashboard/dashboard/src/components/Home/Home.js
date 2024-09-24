import React from 'react';
import { Link } from 'react-router-dom';

function Home() {
  return (
    <div>
      <h1>Welcome to File Classifier</h1>
      <Link to="/classify">Go to File Classifier</Link>
    </div>
  );
}

export default Home;
