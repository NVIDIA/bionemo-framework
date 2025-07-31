import React from 'react';
import './Conclusion.css';

const Conclusion = () => {
  return (
    <section>
      <h1 className="body-header">Looking Forward</h1>
      <p className="body-text">
        Evo 2 represents a leap forward in genomic modeling, capable of
        predicting variant impacts, designing complex DNA sequences, and
        synthesizing biologically meaningful genomes—all without task-specific
        training. Its blend of innovative architecture and vast training data
        heralds a new era of precision genomics, promising breakthroughs in
        personalized medicine, genome editing, and synthetic biology.
        <br />
        <br />
        Still, DNA modeling remains challenging due to its low-entropy
        vocabulary, long-range dependencies spanning millions of bases, and the
        need to capture both local sequence motifs and the broader genomic
        context. Evo 2 makes meaningful progress on these fronts through its
        hybrid architecture and large-scale training. Meanwhile, alternative
        hybrid models such as Evo2-Mamba are also being developed to address
        these challenges, indicating that the best approach for genomic sequence
        modeling may require new architectural innovations beyond standard
        transformers.
        <br />
        <br />
        <i>
          Acknowledgments: We would like to thank Garyk Brixi, Matthew Durrant,
          and Michael Poli for providing relevant data, helpful discussions, and
          feedback on the article.
        </i>
      </p>
    </section>
  );
};

export default Conclusion;
