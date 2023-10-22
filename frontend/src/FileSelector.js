import React, { useState } from 'react';
import { Button } from "reactstrap";
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


function FileSelector({ onFileNameSelected }) {
  
  const [selectedColumn, setSelectedColumn] = useState('');
  const handleFileNameChange = (e) => {
    setSelectedColumn(e.target.value);
  };

  const handleFetchClick = () => {
    // Call the callback function with the selected file name
    console.log('selected column')
    console.log(selectedColumn)
    onFileNameSelected(selectedColumn);
  };
  
  const categrylist = ['먹방', '주식투자', '애견인', '요리', '냥집사', '캠핑', '패션', '홈트레이닝', '룩북', '메이크업', '보디빌딩', 'v-tuber', '베이킹', '게임', '여행&이벤트', '스포츠', '자동차', '코미디', '영화&애니메이션', '과학기술'];

  return (
    <div>
      <h2> 검색하고 싶은 카테고리명을 선택해주세요 </h2>
      <Select value={selectedColumn} onChange={handleFileNameChange}>
        <option value="">Select a column</option>
        {categrylist.map((column) => (
          <option key={column} value={column}>
            {column}
          </option>
        ))}
      </Select>
      {selectedColumn && (
        <p>카테고리 {selectedColumn}이 선택되었습니다.</p>
      )}
      <CusButton color='success' onClick={handleFetchClick}>선택</CusButton>
    </div>
  );
}

export default FileSelector;
