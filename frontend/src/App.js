import React, { useState } from 'react';
import FileSelector from './FileSelector';
import FileDisplay from './FileDisplay';
import TableDisplay from './TableDisplay';
import styled from "styled-components";

const ResultContainer = styled.div`
  color : black;
  padding: 0;
    margin: 0 ;
    display: flex;
    // justify-content:center;
    text-align:center;
    flex-direction: column;
    height: 100vh;
    background-color: white;
`;

function App() {
  const [ColumnData, setColumnData] = useState(null);
  const [RowData,setRowData] = useState(null);
  const [categoryname,setCategoryname] = useState(null);

  const handleFileNameSelected = (categoryName) => {
    // Send a request to Flask to fetch the file by name
    fetch(`/api/files/${categoryName}`)
      .then((response) => {
        if (response.status === 200) {
          return response.text();
        } else {
          throw new Error('File not found');
        }
      })
      .then((data) => {
        // console.log('data',data)
        setRowData('');
        setCategoryname(categoryName);
        setColumnData(data);
      })
      .catch((error) => {
        console.error('Error fetching file:', error);
        setColumnData(null);
      });
  };

  const handleColumnNameSelected = (columnName) => {
    // Send a request to Flask to fetch the file by name
    fetch(`/api/columns/${categoryname}?column=${columnName}`)
      .then((response) => {
        setRowData('');
        if (response.status === 200) {
          return response.text();
        } else {
          throw new Error('File not found');
        }
      })
      .then((data) => {
        // console.log('row',data)
        setRowData(data);
      })
      .catch((error) => {
        console.error('Error fetching file:', error);
        setRowData(null);
      });
  };

  return (
    <ResultContainer className="App">
      <h1>Influencer recommendation</h1>
      <br></br>
      <FileSelector onFileNameSelected={handleFileNameSelected} />
      <br></br>
      {ColumnData && <FileDisplay ColumnData={JSON.parse(ColumnData)} onFileNameSelected={handleColumnNameSelected} />}
      <br></br>
      {RowData && <TableDisplay RowData={JSON.parse(RowData)} />}
    </ResultContainer>
  );
}

export default App;
