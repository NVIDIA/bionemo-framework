import React from 'react';
import './References.css';

const References = () => {
  const references = [
    {
      id: 1,
      title: 'Genome modeling and design across all domains of life with Evo 2',
      authors:
        'Brixi, G., Durrant, M. G., Ku, J., Poli, M., Brockman, G., Chang, D., Gonzalez, G. A., King, S. H., Li, D. B., Merchant, A. T., Naghipourfar, M., Nguyen, E., Ricci-Tam, C., Romero, D. W., Sun, G., Taghibakshi, A., Vorontsov, A., Yang, B., Deng, M., Gorton, L., Nguyen, N., Wang, N. K., Adams, E., Baccus, S. A., Dillmann, S., Ermon, S., Guo, D., Ilango, R., Janik, K., Lu, A. X., Mehta, R., Mofrad, M. R. K., Ng, M. Y., Pannu, J., Ré, C., Schmok, J. C., St. John, J., Sullivan, J., Zhu, K., Zynda, G., Balsam, D., Collison, P., Costa, A. B., Hernandez-Boussard, T., Ho, E., Liu, M. Y., McGrath, T., Powell, K., Burke, D. P., Goodarzi, H., Hsu, P. D. and Hie, B. L., 2025',
      publication: 'bioRxiv, 2025.02.18.638918; doi:10.1101/2025.02.18.638918',
      link: 'https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full',
      hasLink: true,
    },
    {
      id: 2,
      title:
        'Systems and Algorithms for Convolutional Multi-Hybrid Language Models at Scale',
      authors:
        'Jerome Ku, Eric Nguyen, David W. Romero, Garyk Brixi, Brandon Yang, Anton Vorontsov, Ali Taghibakhshi, Amy X. Lu, Dave P. Burke, Greg Brockman, Stefano Massaroli, Christopher Ré, Patrick D. Hsu, Brian L. Hie, Stefano Ermon, Michael Poli',
      link: 'https://arxiv.org/pdf/2503.01868',
      hasLink: true,
    },
    {
      id: 3,
      title: 'Attention is all you need',
      authors:
        'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin',
      publication: 'Advances in Neural Information Processing Systems, 30',
      link: 'https://arxiv.org/abs/1706.03762',
      hasLink: true,
    },
    {
      id: 4,
      title: 'Hyena Hierarchy: Towards Larger Convolutional Language Models',
      authors:
        'Michael Poli*, Stefano Massaroli*, Eric Nguyen*, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon†, Christopher Ré†',
      publication: 'ICML',
      link: 'https://arxiv.org/abs/2302.10866',
      hasLink: true,
    },
    {
      id: 5,
      title:
        'A DNA language model based on multispecies alignment predicts the effects of genome-wide variants',
      authors: 'G. Benegas, C. Albors, A. J. Aw, C. Ye, and Y. S. Song',
      publication: 'Nature Biotechnology',
      hasLink: false,
    },
    {
      id: 6,
      title:
        'Accurate classification of BRCA1 variants with saturation genome editing',
      authors:
        'Gregory M. Findlay, Riza M. Daza, Beth Martin, Melissa D. Zhang, Anh P. Leith, Molly Gasperini, Joseph D. Janizek, Xingfan Huang, Lea M. Starita & Jay Shendure',
      publication: 'Nature',
      hasLink: false,
    },
    {
      id: 7,
      title:
        'Accurate structure prediction of biomolecular interactions with AlphaFold 3',
      authors: 'Abramson, J., Adler, J., Dunger, J. et al.',
      publication: 'Nature',
      link: 'https://www.nature.com/articles/s41586-024-07487-w',
      hasLink: true,
    },
  ];

  return (
    <section className="references-section">
      <div className="references-section-container">
        <h2 className="references-header">References</h2>
        <ol className="references-list">
          {references.map(ref => (
            <li id={`ref-${ref.id}`} key={ref.id} className="reference-item">
              <span className="reference-content">
                <span className="reference-title">{ref.title}</span>
                {ref.link && ref.hasLink && (
                  <a href="#" className="reference-link">
                    {ref.link}
                  </a>
                )}
                <br />
                {ref.authors && (
                  <span className="reference-authors">{ref.authors}. </span>
                )}
                {ref.publication && (
                  <span className="reference-publication">
                    {ref.publication}.{' '}
                  </span>
                )}
                {ref.doi && <span className="reference-doi">{ref.doi}</span>}
              </span>
              <a
                href={`#citation-${ref.id}`}
                className="back-link"
                aria-label="Back to citation"
              >
                ↩︎
              </a>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
};

export default References;
