import React from 'react';
import './Striped.css';
import StripedHyenaBlocks from './StripedHyenaBlocks';
import Evo2Table from '../Evo2Table/Evo2Table';
import SideNote from '../UI/SideNote/SideNote';

const Striped = () => {
  return (
    <>
      <p className="body-text">
        <b>Multi-Hybrid Block Layout</b>
        <br />
        As shown above, the residual block has a multi-hybrid block layout,
        consisting of interleaved hyena and attention operators. The composition
        of this block is a hyperparameter that determines the pattern of
        interleaving hyena and attention operators in the residual block. The
        blocks used across the three Evo 2 model specifications are shown below:
      </p>
      <StripedHyenaBlocks />

      {/* architecture diagram */}
      <div className="diagram-container">
        {/* insert your SE‑MR‑LI diagram here */}
      </div>

      <p className="body-text">
        Although all three model variants employed the same SE-MR-LI-MHA
        multi‑hybrid block layout (though with differing layer counts and
        multi-head attention positions), these aren't the only multi‑hybrid
        block layout configurations possible. Multiple configurations were
        explored. The table below shows perplexity values for four different 7B
        multi-hybrid models:
        <SideNote>
          In particular, these 7B multi-hybrid models were trained on 400B
          tokens of OpenGenome2, with inner filter lengths of 7 for SE and 128
          for MR.
        </SideNote>
      </p>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Layout</th>
              <th>Perplexity</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>MHA‑MHA‑MHA‑MHA</td>
              <td>3.09</td>
            </tr>
            <tr>
              <td>LI‑LI‑LI-MHA</td>
              <td>2.87</td>
            </tr>
            <tr>
              <td>SE‑SE‑LI-MHA</td>
              <td>2.88</td>
            </tr>
            <tr>
              <td>
                <b>SE‑MR‑LI-MHA</b>
              </td>
              <td>
                <b>2.83</b>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <br />
      <p className="body-text">
        The <b>SE-MR-LI-MHA block layout</b> used by Evo 2 yields the best
        perplexity value on pretraining quality. That said, it's interesting to
        note the comparable performance between SE-SE-LI-MHA block
        configurations and LI-LI-LI-MHA pure long convolution layouts,
        suggesting that the former can replace the latter with negligible
        quality loss but significant efficiency gains, a fruitful area for
        future exploration.
      </p>
    </>
  );
};

export default Striped;
