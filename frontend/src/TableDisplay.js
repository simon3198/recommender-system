import React, { Children, Component, useState } from 'react';
import { Row } from 'reactstrap';
import styled from "styled-components";

// import Table from "./components/kit/Table";
const ResultContainer = styled.div`
  color : black;
  padding: 0;
    margin: 0 ;
    display: flex;
    justify-content:center;
    // flex-direction: column;
    height: 100vh;
    background-color: white;
`;
const StyledTable = styled.table`
  border-collapse: collapse;
//   margin-bottom: 20px;
  margin: 10px;

  th, td {
    padding: 8px 4px;
    text-align: left;
    border-bottom: 1px solid #ccc;
  }

  th {
    background-color: #f2f2f2;
  }

  tr:nth-child(even) {
    background-color: #f2f2f2;
  }
`;

const StyledCaption = styled.caption`
  font-weight: bold;
  font-size: 1.7rem;
  color: #333;
  margin: 30px;
  text-align: center;
  margin-bottom: 10px;
`;

function TableDisplay({ RowData }) {
    
    const views = RowData[0]
    const likes = RowData[1]
    const comments = RowData[2]
    const gasungbis = RowData[3]

    // console.log(views)

    const view_data = Object.values(views[0])
    const view_channelid = Object.values(views[1])
    const view_channelname = Object.keys(views[0])

    const view_result = Array.from({ length: view_data.length }, (_, index) => {
        const name = view_channelname[index];
        const value = Math.round(view_data[index]);
        const url = 'https://www.youtube.com/channel/'+view_channelid[index];
      
        // Perform your operation on the values from each list here
        return {name:name, value:value,url:url};
      });

    const like_data = Object.values(likes[0])
    const like_channelid = Object.values(likes[1])
    const like_channelname = Object.keys(likes[0])

    const like_result = Array.from({ length: like_data.length }, (_, index) => {
        const name = like_channelname[index];
        const value = Math.round(like_data[index]);
        const url = 'https://www.youtube.com/channel/'+like_channelid[index];
    
        // Perform your operation on the values from each list here
        return {name:name, value:value,url:url};
    });

    const comment_data = Object.values(comments[0])
    const comment_channelid = Object.values(comments[1])
    const comment_channelname = Object.keys(comments[0])

    const comment_result = Array.from({ length: comment_data.length }, (_, index) => {
        const name = comment_channelname[index];
        const value = Math.round(comment_data[index]);
        const url = 'https://www.youtube.com/channel/'+comment_channelid[index];
        
        // Perform your operation on the values from each list here
        return {name:name, value:value,url:url};
        });
    
    console.log('tttt',gasungbis)
    const gasungbi_data = Object.values(gasungbis[0])
    const gasungbi_channelid = Object.values(gasungbis[1])
    const gasungbi_channelname = Object.keys(gasungbis[0])

    const gasungbi_result = Array.from({ length: gasungbi_data.length }, (_, index) => {
        const name = gasungbi_channelname[index];
        const value = Math.round(gasungbi_data[index]*100)
        const url = 'https://www.youtube.com/channel/'+gasungbi_channelid[index];
        
        // Perform your operation on the values from each list here
        return {name:name, value:value,url:url};
        });

    console.log('gasungbi',gasungbi_result)

    const handleOpenNewTab = (url) => {
        window.open(url, "_blank", "noopener, noreferrer");
      };
      
  return (
    <ResultContainer>
    <StyledTable>
    <thead>
        <StyledCaption>Views</StyledCaption>
      <tr>
        <th>rank</th>
        <th>Name</th>
        <th>Views</th>
        <th>URL</th>
      </tr>
    </thead>
    <tbody>
      {view_result.map((item,index) => (
        <tr key={index}>
          <td>{index+1}</td>
          <td>{item.name}</td>
          <td>{item.value}</td>
          <td><button onClick={()=>handleOpenNewTab(item.url)}>link</button></td>
        </tr>
      ))}
    </tbody>
  </StyledTable>
  <StyledTable>
    <thead>
        <StyledCaption>Likes</StyledCaption>
      <tr>
        <th>Name</th>
        <th>Likes</th>
        <th>URL</th>
      </tr>
    </thead>
    <tbody>
      {like_result.map((item,index) => (
        <tr key={index}>
          <td>{item.name}</td>
          <td>{item.value}</td>
          <td><button onClick={()=>handleOpenNewTab(item.url)}>link</button></td>
        </tr>
      ))}
    </tbody>
  </StyledTable>
  <StyledTable>
    <thead>
        <StyledCaption>Comments</StyledCaption>
      <tr>
        <th>Name</th>
        <th>Comments</th>
        <th>URL</th>
      </tr>
    </thead>
    <tbody>
      {comment_result.map((item,index) => (
        <tr key={index}>
          <td>{item.name}</td>
          <td>{item.value}</td>
          <td><button onClick={()=>handleOpenNewTab(item.url)}>link</button></td>
        </tr>
      ))}
    </tbody>
  </StyledTable>
  <StyledTable>
    <thead>
        <StyledCaption>Effectiveness</StyledCaption>
      <tr>
        <th>Name</th>
        <th>Score</th>
        <th>URL</th>
      </tr>
    </thead>
    <tbody>
      {gasungbi_result.map((item,index) => (
        <tr key={index}>
          <td>{item.name}</td>
          <td>{item.value}</td>
          <td><button onClick={()=>handleOpenNewTab(item.url)}>link</button></td>
        </tr>
      ))}
    </tbody>
  </StyledTable>
  </ResultContainer>
  
    // <div>
    //   <h2>Select a Column</h2>
    //   <select value={selectedColumn} onChange={handleSelectChange}>
    //     <option value="">Select a column</option>
    //     {fileData.map((column) => (
    //       <option key={column} value={column}>
    //         {column}
    //       </option>
    //     ))}
    //   </select>
    //   {selectedColumn && (
    //     <p>You selected: {selectedColumn}</p>
    //   )}
    //   <button onClick={handleFetchClick}>Fetch File</button>
    // </div>
  );
}

export default TableDisplay;
