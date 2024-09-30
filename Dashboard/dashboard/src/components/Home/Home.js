import React from 'react';
import { Link } from 'react-router-dom';

function Home() {
  return (
    <div
      className="h-screen flex items-center justify-center bg-cover bg-center"
      style={{ backgroundImage: 'url(https://www.opensourceforu.com/wp-content/uploads/2020/05/Featured-image-of-python-ML.jpg)', backgroundSize: 'cover', backgroundPosition: 'center' }} // Replace 'your-image.jpg' with the actual image filename
    >
      <div className="absolute inset-0 bg-black opacity-50" /> {/* Adjust opacity as needed */}

      <div className="relative text-center text-white z-10"> {/* z-10 to ensure text is above overlay */}
        <h1 className="text-5xl mb-4 font-extrabold text-black">Welcome to File Classifier</h1>
        <button className='px-4 py-2 rounded bg-red-500 text-black'>
        <Link to="/classify" className="text-lg font-semibold hover:underline">
          Go to File Classifier
        </Link>
        </button>
      </div>
    </div>
  );
}

export default Home;
