// Accordion.jsx
import React, { useState } from 'react';
import './Accordion.css';

const Accordion = ({
  title = 'Relevant Biology Terms',
  open = false,
  children,
}) => {
  const [isOpen, setIsOpen] = useState(open);

  const toggleAccordion = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className="accordion-container body-text">
      <div className="accordion-header" onClick={toggleAccordion}>
        <span className={`accordion-arrow ${isOpen ? 'open' : ''}`}>→</span>
        <h3 className="accordion-title">{title}</h3>
      </div>

      {isOpen && <div className="accordion-content">{children}</div>}
    </div>
  );
};

export default Accordion;
