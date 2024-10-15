import React, { useState } from 'react';
import { FaFolder, FaFile } from "react-icons/fa";
import { CircularProgress } from '@mui/material';

const FileClassifier = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [classifiedFiles, setClassifiedFiles] = useState(null);
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [directory, setDirectory] = useState(null);
  const [isDirectoryMode, setIsDirectoryMode] = useState(true); // State to toggle between Directory and File selection
  const [cacheData, setCacheData] = useState(null); // State for storing cache data


  const handleDirectoryChange = (event) => {
    const files = event.target.files;
    const fileArray = Array.from(files);
    setSelectedFiles(fileArray);
    if (files.length > 0) {
      const path = files[0].webkitRelativePath.split('/')[0];
      setDirectory(path);
    }
    console.log('Selected files:', fileArray);
  };

  const handleFileChange = (event) => {
    const files = event.target.files;
    const fileArray = Array.from(files);
    setSelectedFiles(fileArray);
    console.log('Selected files:', fileArray);
  };

  const handleClassify = async () => {
    if (selectedFiles.length === 0) {
      setSnackbarMessage('Please select files or a directory first.');
      setOpenSnackbar(true);
      return;
    }

    setLoading(true);
    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('http://127.0.0.1:5000/classify-dir', {
        method: 'POST',
        body: formData,
      });

      console.log("Respone: ", response);

      if (!response.ok) {
        const errorData = await response.json();
        setSnackbarMessage(errorData.message || 'Network response was not ok');      
        setOpenSnackbar(true);
        return;
      }

      const data = await response.json();
      console.log("Data: ", data);
      console.log('Classification Result:', data);
      setClassifiedFiles(data);
      setSnackbarMessage('Files classified successfully!');
      setOpenSnackbar(true);
    } catch (error) {
      console.error('Error classifying files:', error);
      setSnackbarMessage('Error classifying files. Please try again.');
      setOpenSnackbar(true);
    } finally {
      setLoading(false);
    }
  };

  const handleSnackbarClose = () => {
    setOpenSnackbar(false);
  };

  // Fetch cache data
  const handleFetchCache = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/display-cache'); // Assuming this is your cache endpoint
      const data = await response.json();
      console.log('Cache Data:', data);
      setCacheData(data.cache_data); // Assuming 'cache_data' is the key in your response
    } catch (error) {
      console.error('Error fetching cache data:', error);
      setSnackbarMessage('Error fetching cache data.');
      setOpenSnackbar(true);
    }
  };

  return (
    <div className="mt-10 p-4">
      <h1 className="text-3xl text-center font-bold mb-4">File Category Classifier</h1>

      {/* Toggle buttons for selecting Directory or Files */}
      <div className="flex justify-center mb-4">
        <button
          className={`px-4 py-2 rounded-l-lg ${isDirectoryMode ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'} transition`}
          onClick={() => setIsDirectoryMode(true)}
        >
          Select Directory
        </button>
        <button
          className={`px-4 py-2 rounded-r-lg ${!isDirectoryMode ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'} transition`}
          onClick={() => setIsDirectoryMode(false)}
        >
          Select Files
        </button>
      </div>

      {/* Directory/File selection area */}
      <div className="flex flex-col items-center gap-4 border py-10 w-1/2 m-auto">
        {isDirectoryMode ? (
          <>
            <button
              className="bg-blue-500 text-white px-4 py-2 rounded flex items-center gap-2"
              type="button"
              onClick={() => document.getElementById('directory-input').click()}
            >
              <FaFolder className="h-6 w-6" />
              Select Directory
            </button>
            <input
              id="directory-input"
              type="file"
              webkitdirectory="true"
              directory="true"
              multiple
              hidden
              onChange={handleDirectoryChange}
            />
            {directory && (
              <p className="mt-2 text-lg">
                Selected Directory: <span className="font-semibold">{directory}</span>
              </p>
            )}
          </>
        ) : (
          <>
            <button
              className="bg-blue-500 text-white px-4 py-2 rounded flex items-center gap-2"
              type="button"
              onClick={() => document.getElementById('file-input').click()}
            >
              <FaFile className="h-6 w-6" />
              Select Files
            </button>
            <input
              id="file-input"
              type="file"
              multiple
              hidden
              onChange={handleFileChange}
            />
            {selectedFiles.length > 0 && (
              <p className="mt-2 text-lg">
                Selected Files: <span className="font-semibold">{selectedFiles.map(file => file.name).join(', ')}</span>
              </p>
            )}
          </>
        )}

        {/* Classify Button */}
        <button
          className={`px-4 py-2 rounded ${!selectedFiles.length || loading ? 'bg-gray-400' : 'bg-green-500'} text-white`}
          onClick={handleClassify}
          disabled={!selectedFiles.length || loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Classify Files'}
        </button>
      </div>

      

      {/* Snackbar for notifications */}
      {openSnackbar && (
        <div className="fixed bottom-4 left-4 bg-teal-200 text-black font-semibold px-4 py-2 rounded text-center">
          <p>{snackbarMessage}</p>
          <button onClick={handleSnackbarClose} className="font-bold text-red-600 text-center text-sm underline mt-2">
            Close
          </button>
        </div>
      )}

      {/* Display Classified Files */}
      {classifiedFiles && (
  <div className="mt-10">
    <h2 className="text-center text-3xl font-bold mb-4">Classified Files</h2>
    <div className="grid grid-cols-1 gap-8 justify-items-center">
      {Object.keys(classifiedFiles).map((category) => {
        const displayCategory = category.includes('Prediction error') || !category ? 'Unknown' : category;
        return (
          <div key={category} className="w-8/12 bg-white shadow-md">
            <div className="flex rounded-md">
              <div className="w-3/12 bg-teal-500 text-white flex items-center justify-center py-11 px-3 text-center rounded-l-lg">
                <h3 className="text-lg font-bold">{displayCategory}</h3>
              </div>
              <div className="ml-4 flex-1 py-7">
                <ul className="list-disc ml-6">
                  {classifiedFiles[category].map((file, index) => (
                    <li key={index} className="flex items-center">
                      <FaFile className="mr-2" />
                      <strong> File : </strong> {file.file}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  </div>
)}

    <div className="mt-6 flex justify-center">
        <button
          className="bg-blue-500 text-white px-4 py-2 rounded"
          onClick={handleFetchCache}
        >
          Fetch Cache Data
        </button>
      </div>

    </div>
  );
};

export default FileClassifier;
