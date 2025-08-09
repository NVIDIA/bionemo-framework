import React from 'react';
import './Title.css';

const Title = () => (
  <section className="title-section">
    <h1 className="title-main">The Illustrated Evo 2</h1>
    <p className="title-sub">
      A visual, interactive walkthrough of Evo 2 and the StripedHyena2
      architecture.
    </p>
    <div className="title-meta">
      <div className="title-meta-content">
        <div className="meta-item">
          <span className="meta-label">Authors</span>
          <p>Jared Wilber</p>
          <p>Farhad Ramezanghorbani</p>
          <p>David Romero Guzman</p>
          <p>Tyler Shimko</p>
          <p>John St. John</p>
        </div>
        <div className="meta-item">
          <span className="meta-label">Affiliations</span>
          <p>NVIDIA</p>
          <p>NVIDIA</p>
          <p>NVIDIA</p>
          <p>NVIDIA</p>
          <p>NVIDIA</p>
        </div>
        <div className="meta-item">
          <span className="meta-label">Published</span>
          <p>July 30, 2025</p>
        </div>
      </div>
    </div>
  </section>
);

export default Title;
