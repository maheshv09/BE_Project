import React, { useState } from 'react';
import { FaFolder } from "react-icons/fa";
import { FaFile } from "react-icons/fa";
import { CircularProgress } from '@mui/material';  // Retain CircularProgress if needed, or create a custom one

const FileClassifier = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [classifiedFiles, setClassifiedFiles] = useState(null);
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [directory, setDirectory] = useState(null);

  const handleDirectoryChange = (event) => {
    const files = event.target.files;
    const fileArray = Array.from(files);  // Convert FileList to an array
    setSelectedFiles(fileArray);
    if (files.length > 0) {
      const path = files[0].webkitRelativePath.split('/')[0];  // Get directory name
      setDirectory(path);  // Set the directory name
    }
    console.log('Selected files:', fileArray);
  };

  const handleClassify = async () => {
    if (selectedFiles.length === 0) {
      setSnackbarMessage('Please select a directory first.');
      setOpenSnackbar(true);
      return;
    }

    setLoading(true);

    // Prepare FormData to send files to the server
    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('http://localhost:5000/classify-dir', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      console.log('Classification Result:', data);
      setClassifiedFiles(data);  // Store API response (classified categories and files)
      setSnackbarMessage('Files classified successfully !');
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

  return (
    <div className=" mt-10 p-4">
      <h1 className="text-3xl text-center font-bold mb-4">File Category Classifier</h1>

      {/* Directory and Classify buttons stacked vertically */}
      <div className="flex flex-col items-center gap-4 border py-10 w-1/2 m-auto">
        {/* Directory Select Button */}
        <button
          className="bg-blue-500 text-white px-4 py-2 rounded flex items-center gap-2"
          type="button"
          onClick={() => document.getElementById('file-input').click()}
        >
          <FaFolder className="h-6 w-6" />
          Select Directory
        </button>
        <input
          id="file-input"
          type="file"
          webkitdirectory="true"
          directory="true"
          multiple
          hidden
          onChange={handleDirectoryChange}
        />

        {/* Display the selected directory */}
        {directory && (
          <p className="mt-2 text-lg">
            Selected Directory: <span className="font-semibold">{directory}</span>
          </p>
        )}

        {/* Classify Button - Disabled until directory is selected */}
        <button
          className={`px-4 py-2 rounded ${!directory || loading ? 'bg-gray-400' : 'bg-green-500'} text-white`}
          onClick={handleClassify}
          disabled={!directory || loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Classify Files'}
        </button>
      </div>

      {/* Display Snackbar for notifications */}
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
          <h2 className=" text-center text-3xl font-bold mb-4">Classified Files</h2>
          <div className="grid grid-cols-1 gap-8 justify-items-center">
            {Object.keys(classifiedFiles).map((category) => (
              <div key={category} className="w-8/12 bg-white shadow-md ">
                <div className="flex rounded-md">
                  {/* Category in teal box */}
                  <div className="w-3/12 bg-teal-500 text-white flex items-center justify-center py-11 px-3 text-center rounded-l-lg">
                    <h3 className="text-lg font-bold">{category}</h3>
                  </div>

                  <div className="ml-4 flex-1 py-7">
                    <ul className="list-disc ml-6">
                      {classifiedFiles[category].map((file, index) => (
                        <li key={index} className="flex items-center">
                          <FaFile className="mr-2" />
                          <strong>File:</strong> {file.file} <br />
                          {/* <strong>MIME Type:</strong> {file.mime_type} */}
                        </li>
                      ))}
                    </ul>
                  </div>


                </div>
              </div>

            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default FileClassifier;
