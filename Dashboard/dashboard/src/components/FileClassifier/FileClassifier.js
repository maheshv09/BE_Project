import React, { useState } from 'react';
import { Button, Typography, Box, CircularProgress, Snackbar, Card, CardContent, Grid, Alert } from '@mui/material';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import MuiAlert from '@mui/material/Alert';

const FileClassifier = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [classifiedFiles, setClassifiedFiles] = useState(null);
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [directory, setDirectory] = useState(null);
  const handleDirectoryChange = (event) => {
    const files = event.target.files;
    const fileArray = Array.from(files); // Convert FileList to an array
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

  return (
    <Box sx={{ textAlign: 'center', mt: 5, p: 2 }}>
  <Typography variant="h4" gutterBottom>
    File Category Classifier
  </Typography>

  {/* Directory and Classify buttons stacked vertically */}
  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
    {/* Directory Select Button */}
    <Button
      variant="contained"
      component="label"
      startIcon={<FolderOpenIcon />}
      sx={{ width: 'fit-content', mb: 2 }}  // Ensure button has margin below it
    >
      Select Directory
      <input
        type="file"
        webkitdirectory="true"
        directory="true"
        multiple
        hidden
        onChange={handleDirectoryChange}
      />
    </Button>

    {/* Display the selected directory */}
    {directory && (
      <Typography variant="subtitle1" sx={{ mt: 1 }}>
        Selected Directory: {directory}
      </Typography>
    )}

    {/* Classify Button - Disabled until directory is selected */}
    <Button
      variant="contained"
      color="primary"
      onClick={handleClassify}
      sx={{ width: 'fit-content', mb: 2 }}  // Ensure button has margin below it
      disabled={!directory || loading}  // Disable if no directory selected or during loading
    >
      {loading ? <CircularProgress size={24} /> : 'Classify Files'}
    </Button>
  </Box>

  {/* Display Snackbar for notifications */}
  <Snackbar open={openSnackbar} autoHideDuration={6000} onClose={handleSnackbarClose}>
    <Alert onClose={handleSnackbarClose} severity="info" sx={{ width: '100%' }}>
      {snackbarMessage}
    </Alert>
  </Snackbar>

  {/* Display Classified Files */}
  {classifiedFiles && (
    <Box sx={{ mt: 5 }}>
      <Typography variant="h5" gutterBottom>
        Classified Files
      </Typography>
      <Grid container spacing={2} justifyContent="center">
        {Object.keys(classifiedFiles).map((category) => (
          <Grid item xs={12} sm={6} md={4} key={category}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {category}
                </Typography>
                <ul>
                  {classifiedFiles[category].map((file, index) => (
                    <li key={index}>
                      <strong>File:</strong> {file.file} <br />
                      <strong>MIME Type:</strong> {file.mime_type}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  )}
</Box>

  );
};

export default FileClassifier;
