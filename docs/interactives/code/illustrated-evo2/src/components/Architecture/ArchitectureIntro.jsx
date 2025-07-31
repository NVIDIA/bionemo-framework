import React from 'react';

import ArchitectureCard from './ArchitectureCard';
import Evo2Table from '../Evo2Table/Evo2Table';

const ArchitectureIntro = () => {
  return (
    <>
      <h1 className="body-header">Architecture</h1>
      <p className="body-text">
        Evo 2 is based on the StripedHyena2 architecture, an autoregressive
        hybrid architecture formed by convolutional and multi-head attention
        layers designed for next-token prediction that follows the same form as
        GPT-style transformer models.
        <a id="citation-3" href="#ref-3" className="citation-link">
          <sup>3</sup>
        </a>
        <br />
        <br />
        Byte-tokenized DNA is embedded before flowing through the residual
        block, with RMSNorms (pre-)positioned throughout. The residual block
        itself consists of StripedHyena2 layers, followed by a feedforward
        layer. Each <b>StripedHyena2 Layer</b> contains either one of the three
        hyena operators or a multi-head attention operator. In aggregate, these
        interleaved hyena and attention operators comprise the multi-hybrid
        block layout, the output of which is then passed through a final layer
        norm and linear output layer:
      </p>
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <ArchitectureCard />
      </div>
      <br />
      <br />
      <p className="body-text">
        The table below summarizes the total tokens processed, parameter counts,
        and additional architecture details for each Evo 2 model specification:
      </p>
      <Evo2Table />
    </>
  );
};

export default ArchitectureIntro;
