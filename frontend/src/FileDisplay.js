import React, { useState } from 'react';
import styled from "styled-components";

export const Select = styled.select`
	margin: 0 auto;
	min-width: 0;
	display: block;
  text-align : center;
  width : 200px;
	padding: 8px 8px;
	font-size: inherit;
	line-height: inherit;
	border: 1px solid;
	border-radius: 4px;
	color: inherit;
	background-color: transparent;
	&:focus {
		border-color: red;
	}
`;

const CusButton = styled.button`

margin: 2px;
border: none;
cursor: pointer;
font-family: "Noto Sans KR", sans-serif;
font-size: var(--button-font-size, 1rem);
padding: var(--button-padding, 8px 16px);
border-radius: var(--button-radius, 8px);
background: var(--button-bg-color, #0d6efd);
color: var(--button-color, #ffffff);

&:hover{
  background: var(--button-hover-bg-color, #025ce2);
}

`;

function FileDisplay({ ColumnData,onFileNameSelected }) {
    const [columns, setColumns] = useState([]);
    const [selectedColumn, setSelectedColumn] = useState('');
    console.log('yes')
    console.log(ColumnData)
    const data = '';
    const handleSelectChange = (e) => {
        setSelectedColumn(e.target.value);
      };

      const handleFetchClick = () => {
        // Call the callback function with the selected file name
        console.log(selectedColumn)
        onFileNameSelected(selectedColumn);
      };
  return (
    <div>
      <h2>검색하고 싶은 키워드를 선택해주세요</h2>
      <Select value={selectedColumn} onChange={handleSelectChange}>
        <option value="">Select a column</option>
        {ColumnData.map((column) => (
          <option key={column} value={column}>
            {column}
          </option>
        ))}
      </Select>
      {selectedColumn && (
        <p>키워드 {selectedColumn}이 선택되었습니다.</p>
      )}
      <CusButton onClick={handleFetchClick}>선택</CusButton>
    </div>
  );
}

export default FileDisplay;
