import React from 'react';
import './Intro.css';
import SideNote from '../UI/SideNote/SideNote';
import Accordion from '../UI/Accordion/Accordion';

const Intro = () => {
  return (
    <section>
      <p className="body-text">
        Evo 2
        <a id="citation-1" href="#ref-1" className="citation-link">
          <sup>1</sup>
        </a>{' '}
        marks the largest genomic modeling effort to date, trained on over 8.8
        trillion nucleotides from bacteria, archaea, eukarya, and bacteriophage.
        Built on the StripedHyena2
        <a id="citation-2" href="#ref-2" className="citation-link">
          <sup>2</sup>
        </a>{' '}
        architecture—a hybrid of convolutional and self-attention layers—the
        model combines Hyena operators with multi-head attention to capture both
        local functional features and long-range genomic dependencies. In this
        article, we provide a deep, yet intuitive walkthrough of Evo 2's data
        and training process, the underlying architecture, and how the model can
        be used for genetic experiments without task-specific fine-tuning.
        <br />
        <br />
        If you don't have a background in biology, don't worry - throughout the
        article, we provide dropdowns like the one below, detailing any
        prerequisite or relevant biology knowledge. Simply click on the dropdown
        to reveal (or hide) the information.
        <SideNote>
          If you already have a genetics background, feel free to skip the
          dropdowns.
        </SideNote>
      </p>

      <div className="body-text">
        <Accordion title="Core Genetics Concepts" open={false}>
          <p className="body-text">
            <b>DNA</b>: Deoxyribonucleic acid is the hereditary molecule
            containing genetic instructions for development, functioning, and
            reproduction, typically arranged as a double helix composed of
            nucleotide base pairs.
          </p>
          <p className="body-text">
            <b>RNA</b>: Ribonucleic acid is a single-stranded nucleic acid that
            acts as a messenger carrying instructions from DNA for controlling
            protein synthesis and performs crucial roles in coding, decoding,
            and regulating genetic information.
          </p>
          <p className="body-text">
            <b>Proteins</b>: Complex molecules composed of amino acid chains
            that perform a vast array of functions within organisms, including
            catalyzing metabolic reactions, DNA replication, responding to
            stimuli, and transporting molecules.
          </p>
          <p className="body-text">
            <b>Nucleotides</b>: The basic building blocks of DNA and RNA,
            consisting of a sugar, phosphate group, and a nitrogenous base.
          </p>
          <p className="body-text">
            <b>Nucleotide Sequences</b>: The specific order of nucleotides
            (adenine, guanine, cytosine, thymine/uracil) in a DNA or RNA
            molecule that encodes genetic information and determines the amino
            acid sequence in proteins.
          </p>
          <p className="body-text">
            <b>The Central Dogma</b>: The fundamental principle in molecular
            biology describing the flow of genetic information from DNA to RNA
            to protein, establishing the directional transfer of biological
            information. (This is what allows a DNA model to be used to generate
            proteins.)
          </p>
          <p className="body-text"></p>
        </Accordion>
      </div>
      <p className="body-text">Let's get started!</p>
    </section>
  );
};

export default Intro;
