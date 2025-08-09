import React from 'react';
import Title from './components/Title/Title';
import Intro from './components/Intro/Intro';
import References from './components/References/References';
import Striped from './components/Striped/Striped';
import HyenaIntro from './components/HyenaIntro/HyenaIntro';
import Conclusion from './components/Conclusion/Conclusion';
import HyenaHighlight from './components/HyenaHighlight/HyenaHighlight';
import Loss from './components/Loss/Loss';
import Data from './components/Data/Data';
import Experiments from './components/Experiments/Experiments';
import BRCA1 from './components/BRCA1/BRCA1';
import ArchitectureIntro from './components/Architecture/ArchitectureIntro';
import GenomeGeneration from './components/GenomeGeneration/GenomeGeneration';
import ChromatinScroll from './components/GenomeGeneration/ChromatinScroll';
import './App.css';

const App = () => {
  return (
    <div className="App">
      <Title />
      <Intro />
      <Data />
      <ArchitectureIntro />
      <HyenaIntro />
      <HyenaHighlight />
      <Striped />
      <Loss />
      <Experiments />
      <BRCA1 />
      <GenomeGeneration />
      <ChromatinScroll />
      <Conclusion />
      <References />
    </div>
  );
};

export default App;
