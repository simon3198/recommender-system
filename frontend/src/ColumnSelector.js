import React, { useState, useEffect } from 'react';

function ColumnSelector() {
  const [columns, setColumns] = useState([]);
  const [selectedColumn, setSelectedColumn] = useState('');

  useEffect(() => {
    // Fetch the list of columns from the Flask server
    fetch('/api/get_columns')
      .then((response) => response.json())
      .then((data) => {
        setColumns(data);
      })
      .catch((error) => {
        console.error('Error fetching columns:', error);
      });
  }, []);

  const handleSelectChange = (e) => {
    setSelectedColumn(e.target.value);
  };

  return (
    <div>
      <h2>Select a Column</h2>
      <select value={selectedColumn} onChange={handleSelectChange}>
        <option value="">Select a column</option>
        {columns.map((column) => (
          <option key={column} value={column}>
            {column}
          </option>
        ))}
      </select>
      {selectedColumn && (
        <p>You selected: {selectedColumn}</p>
      )}
    </div>
  );
}

export default ColumnSelector;
