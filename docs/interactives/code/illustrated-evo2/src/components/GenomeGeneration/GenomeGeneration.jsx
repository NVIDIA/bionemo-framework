import React from 'react';
import GenomeGenerationTable from './GenomeGenerationTable.jsx';
import Accordion from '../UI/Accordion/Accordion';
import SideNote from '../UI/SideNote/SideNote';
// import BeamSearch from "./BeamSearchH";
import BeamSearchSteps from './BeamSearchSteps/BeamSearchSteps';
import DesignBar from './DesignBar.jsx';
import SparkDesignBar from './SparkDesignBar.jsx';

const GenomeGeneration = () => {
  return (
    <>
      <h1 className="body-header">Genome Generation</h1>

      <section>
        <p className="body-text">
          Evo 2 is capable of autoregressive generation of DNA sequences one
          base at a time, using its learned representations. This can be done in
          an unguided manner (<b>zero-shot generation</b>) to produce novel
          sequences, or in a goal-directed way (<b>inference-time compute</b>)
          to search for sequences that satisfy specific criteria.
        </p>
      </section>

      <section className="mb-8">
        <p className="body-text">
          <b>Zero-Shot Generation</b>
          <br />
          To generate genomes in a zero-shot manner, one can simply prompt Evo2
          with the beginning of a sequence (e.g. <i>ATCG...</i>) and let it
          generate the rest.
          <br />
          <br />
          <a
            style={{ color: 'var(--mine-shaft)', fontWeight: 'bold' }}
            href="https://build.nvidia.com/arc/evo2-40b"
            target="_blank"
            rel="noopener noreferrer"
          >
            [Note: Click here to interactively sample Evo 2's genome generation
            capabilities.]
          </a>
          <br />
          <br />
          To showcase Evo 2's ability to generate <i>novel</i> genomic sequences
          and demonstrate its strong generalization, the Evo 2 40B model was
          prompted with a human mitochondrial sequence as a starting point to
          create a diverse collection of "viable" eukaryotic genomes. Two
          hundred and fifty unique mitochondrial genomes were generated, and
          then compared to different organisms using{' '}
          <a
            href="https://blast.ncbi.nlm.nih.gov/Blast.cgi"
            target="_blank"
            rel="noopener noreferrer"
          >
            BLASTp
          </a>
          . That is, three core proteins were selected from the generated
          genomes and compared to other organisms. These core protein results
          were then validated with AlphaFold 3
          <a id="citation-7" href="#ref-7" className="citation-link">
            <sup>7</sup>
          </a>
          , where the generated structures matched the respective mitochondrial
          proteins.
          <SideNote>
            The generated proteins were highly similar to their natural analogs,
            achieving pLDDT scores between 0.67 and 0.83.
          </SideNote>
        </p>
        <p className="body-text">
          A subset of the generated genomes, all seeded from the same human
          mitochondrial sequence, is shown below:
        </p>
        <GenomeGenerationTable />
        <br />
        <p className="body-text">
          While the generated genomes show impressive variety—some closely
          resemble sheep, while others are most similar to fish—many genome
          design tasks require more than just diversity. They demand that
          generated sequences satisfy specific biological criteria.
        </p>

        <p className="body-text">
          {/* <b>Inference-Time Compute</b> */}
          {/* <br /> */}
          {/* By taking advantage of Inference-Time Compute, Evo 2 can move beyond unconstrained autoregressive generation
          to generate sequences that satisfy specific biological criteria. */}
        </p>
        <p className="body-text">
          <b>Guided Genome Generation</b>
          <br />
          While Evo 2 is powerful enough to create novel genomes simply by
          extending an initial sequence, researchers often want more precise
          control—generating DNA sequences tailored to specific biological goals
          or desired genomic features. By taking advantage of inference-time
          compute, Evo 2 can move beyond unconstrained autoregressive generation
          to generate sequences that satisfy specific biological criteria.
          <SideNote>
            Inference-time compute refers to the computational resources
            invested when using a trained model to produce outputs or
            predictions. This is actually the first inference-time scaling
            result for biological language modeling.
          </SideNote>
          Rather than generating sequences at random, Evo 2's inference is
          updated to allow for predicting and evaluating multiple potential DNA
          sequences at each step to identify the best matches for specific
          biological objectives.
        </p>
        <BeamSearchSteps />
        <p className="body-text">
          <b>Beam Search for Guided Chromatin Accessibility Design</b>
          <br />
          As a case study motivating the ability to guide genome generation
          according to some specific functional goal, let's walk through guiding
          Evo 2 with beam search to generate a DNA sequence with a desired
          chromatin accessibility pattern.
          <br />
        </p>
        <p className="body-text">
          <Accordion title="Chromatin Accessibility" open={true}>
            <p className="body-text">
              Chromatin accessibility is a key factor in gene expression
              regulation. It refers to the openness or compactness of chromatin,
              which in turn controls which DNA regions can be accessed by
              transcriptional machinery. Chromatin accessibility is modulated by
              both DNA sequence and epigenetic modifications, such as histone
              modifications and DNA methylation. Chromatin accessibility refers
              to how open or closed certain regions of DNA are, influencing
              whether genes in those regions can be actively expressed.
            </p>
          </Accordion>
        </p>
        <p className="body-text">
          To begin, a binary chromatin accessibility pattern was designed to
          maximize accessibility in certain genomic regions while minimizing it
          in others:
        </p>
        <div
          style={{
            margin: '1rem auto',
            maxWidth: '400px',
            display: 'flex',
            justifyContent: 'center',
          }}
        >
          <DesignBar
            dnaSequence="ATCGTCAGCGATTGCGTACG"
            peakPattern={
              // 1 if base is T or G, else 0
              Array.from('ATCGTCAGCGATTGCGTACG').map(b =>
                b === 'C' || b === 'G' ? 1 : 0
              )
            }
            height={70}
            strokeWidth={3}
            width={400}
          />
        </div>
        <p className="body-text">
          Then, the goal is to guide Evo 2 to generate DNA sequences that have
          the above chromatin accessibility pattern,{' '}
          <SparkDesignBar strokeWidth={2} dnaSequence="ATCGTTA" />, while
          matching the desired peak pattern,{' '}
          <SparkDesignBar
            dnaSequence=""
            strokeWidth={2}
            peakPattern={[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0]}
          />
          , via inference-time compute with beam search.
          <br />
          <br />
          Let's walk through the process of guiding Evo 2 to generate a DNA
          sequence with the desired chromatin accessibility pattern
          step-by-step:
        </p>
        {/* <p className="body-text">
          Instead of operating at the nucleotide level, Evo 2 autoregressively generates multiple possible next segments of DNA. Each of these candidate sequences is then scored based on predictions from external models—in this experiment, using Enformer and Borzoi—to estimate how they affect the chromatin accessibility of the genome. 
          Evo 2 compares the predicted accessibility of each candidate against a carefully designed target pattern. At each generation step, the sequences that best match this desired accessibility pattern are retained and extended further. By repeating this guided generation process iteratively, Evo 2 constructs entire DNA sequences whose predicted biological behavior closely adheres to the researchers' predefined goals.
        </p> */}
      </section>
    </>
  );
};

export default GenomeGeneration;
