import React from 'react';
import katexify from '../../utils/katexify';
import './Loss.css';
import RepetitionChart from './RepetitionChart';
import Accordion from '../UI/Accordion/Accordion';

const Loss = () => {
  const formula = String.raw`
  \ell_{wCE} \;=\; \frac{1}{Z}\sum_{t} w_t\,\ell_{CE}(t)
  \\\\
  w_t = \begin{cases}
    0.1, & \text{if position } t \text{ is in a repetitive region}\\[0.5em]
    1.0, & \text{otherwise}
  \end{cases}
  \\\\
`;

  return (
    <>
      <h1 className="body-header">Loss</h1>

      <p className="body-text">
        Given the autoregressive, next-token prediction formulation of Evo 2, it
        should come as no surprise that the model is optimized with a
        cross-entropy loss. Following the lead of other DNA models
        <a id="citation-5" href="#ref-5" className="citation-link">
          <sup>5</sup>
        </a>
        , Evo 2 is trained with a modified cross-entropy that reweights the loss
        contribution of repetitive portions of DNA by 0.1:
      </p>
      <div className="loss-formula">
        <span
          dangerouslySetInnerHTML={{
            __html: katexify(formula, true),
          }}
        />
      </div>

      <p className="body-text">
        where{' '}
        <span
          dangerouslySetInnerHTML={{
            __html: katexify(
              'Z = 0.1\\,N_{\\mathrm{repeat}} + N_{\\mathrm{non\\_repeat}}'
            ),
          }}
        />{' '}
        is the normalization factor that ensures consistent loss scaling
        regardless of the proportion of repetitive regions,{' '}
        <span
          dangerouslySetInnerHTML={{
            __html: katexify('w_t'),
          }}
        />{' '}
        is the weight applied to each position,{' '}
        <span
          dangerouslySetInnerHTML={{
            __html: katexify('N_{\\mathrm{repeat}}'),
          }}
        />{' '}
        represents the number of positions in repetitive regions within a batch,
        and{' '}
        <span
          dangerouslySetInnerHTML={{
            __html: katexify('N_{\\mathrm{non\\_repeat}}'),
          }}
        />{' '}
        is the number of non-repetitive positions. Finally,{' '}
        <span
          dangerouslySetInnerHTML={{
            __html: katexify('Z'),
          }}
        />{' '}
        is the normalization factor that ensures consistent loss scaling
        regardless of the proportion of repetitive regions.
        <br />
        <br />
        Repetitive regions—such as satellite DNA—are abundant in many genomes
        but typically contain less functional information.
      </p>

      <Accordion
        title="Repetitive Regions in DNA"
        open={true}
        className="chart-container body-text"
      >
        <div className="loss-chart-section">
          <p className="loss-chart-text">
            Repeat patterns in the DNA are more challenging to both sequence,
            analyze, and model. In Evo 2,{' '}
            <span className="non-repetitive">non-repetitive</span> DNA are
            weighted normally at <b>1.0</b>, while portions that are{' '}
            <span className="repetitive">repetitive</span> are down-weighted
            using a weight of <b>0.1</b>, so that they contribute less to the
            model's overall loss.
            <br />
            Repetition occurs in DNA in many different ways:
            <br />
            <br />
            Short DNA motifs that repeat directly adjacent to one another, known
            as <b>tandem repeats</b>:
          </p>
          <RepetitionChart
            sequence="ACGTAGCTATATATATATCGGATCGATCGAATATCGACGT"
            repeatPattern="8-17"
          />
        </div>
        <div className="loss-chart-section">
          <p className="loss-chart-text">
            Repeated three-nucleotide sequences are called{' '}
            <b>trinucleotide repeats</b>:
          </p>
          <RepetitionChart
            sequence="GCTCGATACAGCAGCAGCAGCAGCAGCAGATCGATCGGATCGATCG"
            repeatPattern="9-29"
          />
        </div>
        <div className="loss-chart-section">
          <p className="loss-chart-text">
            <b>Transposable elements</b> are DNA sequences that move from one
            location on the genome to another.
          </p>
          <RepetitionChart
            // 12 bases: ACGTACGATGGC ... 12 random ... 12 bases: ACGTACGATGGC
            sequence="ACGTACGATGGCATCGTACGATCGACGTACGATGGC"
            repeatPattern="1-12,25-36"
          />
        </div>
        <div className="loss-chart-section">
          <p className="loss-chart-text">
            Sections with simple sequence patterns are called{' '}
            <b>low complexity regions</b>:
          </p>
          <RepetitionChart
            sequence="ACGTACGATGGGGGGGGGGCTACTATCGATCGATCGATCGATCGAT"
            repeatPattern="10-19"
          />
        </div>
        <div style={{ marginTop: '1.5em' }}>
          <p className="loss-chart-text">
            Other examples of DNA repetition include microsatellites,
            minisatellites, large segmental duplications, and even whole genome
            duplications.
          </p>
        </div>
      </Accordion>

      <p className="body-text">
        The model was evaluated during training using perplexity, as well as a
        number of downstream experiments, discussed in the following section.
      </p>
    </>
  );
};

export default Loss;
