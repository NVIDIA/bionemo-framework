# BioNeMo Service Playground

The BioNeMo Service provides a web-based playground that offers a quick and convenient way to try the BioNeMo models. When you manually enter a prompt or select a preset and hit either *Generate*, the service responds with an embedding of the sequence, generated sequences, embeddings of molecules, generated molecules, a predicted 3D structure, or a proposed docking configuration, depending on the model selected. You can specify which model to submit your request to change the model.

![Screen shot of BioNeMo playground featuring AlphaFold-2](images/image1.png)

Figure 1. An example of the BioNeMo playground featuring AlphaFold-2

As shown in the previous screen, BioNeMo Service provides tooltips that make navigating the UI easier. Hovering over the information icons displays the tooltips, while selecting the *Learn More* button provides brief information regarding models.

![Screenshot of learning more about models](images/image2.png)

Figure 2. Learning about models

## Using the Playground

When you first access the playground, a screen similar to the following displays.

![Screenshot of BioNeMo first experience](images/image3.png)

Figure 3. BioNeMo Service first experience

### Pre-trained Models

On the left side of the screen, BioNeMo Service provides nine pre-trained models for use:

* AlphaFold-2
* OpenFold
* ESMFold
* ESM-1nv
* ESM-2
* MegaMolBART
* MoFlow
* ProtGPT2
* DiffDock

The AlphaFold-2, OpenFold, and ESMFold models are designed to predict the 3-dimensional structure of a protein from sequence alone. This long-standing grand challenge in structural biology reached a significant milestone at CASP14, where AlphaFold-2 achieved nearly experimental accuracy for predicted structures.

Embeddings from ESM-1nv and ESM-2 can be used to fit downstream task models for protein properties of interest. This is accomplished by training a typically much smaller model with a supervised learning objective to infer a property from ESM embeddings of protein sequences.

The embeddings from the MegaMolBART encoder can be used as features for predictive models. Alternatively, the encoder and decoder can be used together to generate novel molecules by sampling the model's latent space starting from a seed molecule. MoFlow can also be used to generate molecules starting from a seed.

ProtGPT2 can be used for unconstrained protein sequence generation by providing a prompt. DiffDock will identify a binding pocket and the best ligand poses in a user provided protein and dock a ligand starting from a user supplied ligand and protein.

### Example Use Cases

On the other left side of the screen, you can also find the available use cases. BioNeMo Service use cases provide the following:

* Protein structure prediction
* Downstream task models for proteins
* Downstream task models for molecules
* Generation of molecules
* Docking a ligand in a protein

#### Downstream Task Models for Proteins

Embeddings from ESM-1nv or ESM-2 can be used to fit downstream task models for protein properties of interest. This is accomplished by training a typically much smaller model with a supervised learning objective to infer a property from ESM-1nv or ESM-2 embeddings of protein sequences.  Using embeddings from these models typically results in far superior accuracy in the final model.

#### Downstream Task Models for Molecules

Embeddings from MegaMolBART can be used as features for predictive models. Like those for proteins, embeddings of small molecules, also known as learned representations, can be used to efficiently train downstream task models.  For example, an XGBoost model could be trained to predict solubility or toxicity.

#### Generation of Molecules

Using MegaMolBART's encoder and decoder, novel molecules can be produced by sampling randomly from a fixed radius in the model's latent space. Users can enter either a list of molecules as SMILES and then select the number of molecules to be generated and the radius for sampling. MoFlow accepts a single SMILES and the number of molecules to generate.

##### Ligand in a Protein

DiffDock is a state-of-the-art generative model for blind molecular docking pose estimation. It takes protein and molecule 3D structures as input, and does not require any information about a binding pocket. It outputs many sampled docking poses through a probabilistic reverse diffusion process and it ranks the generated poses via its confidence model.

#### Protein Generation

##### Inputs

Navigate to the Protein Generation tab to generate novel proteins from the LLM service. There should only be “ProtGPT2” available under the *Model* drop-down. The various parameters affect the generated outputs, including the number of sequences returned and various quality filters. Both sliders and editing the text fields can be used to modify input parameters.

![Screenshot of landing page for protein generation](images/image4.png)

Figure 4. Landing page for protein generation

Use the green *Generate* button on the bottom to kick off generation, where you should see a spinning Generate logo. From here, click **Stop** to terminate the generation early. For example, ‘Return Sequences’ is set to the maximum value, there can be a longer delay than expected to receive a result. When generation is complete, there should be two available options. The first downloads an associated FASTA file, with one entry for each generated protein. FASTA entries are represented first by a sequence ID, that begins with ‘>’. In this case they should contain the length and perplexity of the sequence in the ID. This is followed by the sequence itself.

##### Outputs

Download the file and open it with Notepad or any other text editor to verify. The downloaded result is in the FASTA format. Alternatively, move to our example container and explore how to perform the same computation by using our API.

![Screenshot of example FASTA file returned by ProtGPT](images/image5.png)

Figure 5. Example FASTA file returned by ProtGPT

![Example input parameters for ProtGPT](images/image6.png)

Figure 6. Protein generation output download

The second option on the right-hand column is to open an example container, which should navigate to the NGC catalog.

#### Protein Embedding

##### Inputs

First, select the model of your choice on the top *Model* drop-down menu. All of the options return an embedding for the given sequences. In this example, we will use ESM1-nv. Next there are two options, let BioNeMo fetch a UniProt sequence for you by entering the ID into the second box on the left, or to manually insert the sequence yourself. Alternatively, use one of our pre-selected examples for ultra fast execution time.

![BioNeMo screenshot](images/image7.png)

Figure 7. Protein embedding parameters

##### Outputs

Outputs are returned as binary embeddings, storied as npz files. Select the download button to download them locally, and rely on numpy (`numpy.load`) to load the files locally. Alternatively, you can explore our example container to see how you can incorporate the BioNeMo service API into your workflows.

![BioNeMo screenshot](images/image8.png)

Figure 8. Protein embedding output download

#### Molecule Generation

##### Inputs
You can select between two models for molecule generation, MegaMolBART or MoFlow. Enter a Compound/Chemical id from PubChem to lookup a SMILES, or enter your own custom SMILES in the box below.

![BioNeMo screenshot](images/image9.png)

Figure 9. Molecular generation parameters

![BioNeMo screenshot](images/image10.png)

Figure 10. Molecular generation input

Alternatively, use one of our pre-selected examples for fast generation. In this example, we select Aspirin.

##### Outputs

When generation is complete, you should see a visualization of the generated SMILES with a color-coded corresponding Tanimoto similarity score.

![BioNeMo screenshot](images/image11.png)

Figure 11. Molecular generation output

![BioNeMo screenshot](images/image12.png)

Figure 12. Molecule generation SMILES output

Select the download button to download the SMILES and similarity scores as a CSV file.

#### Molecule Embedding

##### Inputs

You can select between two models for molecule generation, MegaMolBART or MoFlow. Enter a Compound/Chemical id from PubChem to lookup a SMILES or enter your own custom SMILES in the box below.

![BioNeMo screenshot](images/image13.png)

Figure 13. Molecule embedding parameters

Alternatively, use one of our pre-selected examples for fast embedding. In this example, we select Aspirin.

![BioNeMo screenshot](images/image14.png)

Figure 14. Molecule embedding input

##### Outputs

Outputs are returned as binary embeddings, stored as npz files. Select the download button to download them locally, and rely on numpy (`numpy.load`) to load the files locally. Alternatively, you can explore our example container to see how you can incorporate the BioNeMo service API into your workflows.

![BioNeMo screenshot](images/image15.png)

Figure 15. Molecule embedding output


#### Protein Structure Prediction

##### Inputs

To generate a protein structure, navigate to the *Protein Folding* section, where you will be presented with the following prompt:

![BioNeMo screenshot](images/image16.png)

Figure 16. Protein folding parameters

First select the model of your choice on the top *Model* drop-down menu. All of the options return a folded protein. In this example, we will use AlphaFold-2. Next, there are two options; let BioNeMo fetch a Uniprot sequence for you by entering the ID into the second box on the left or to  insert the sequence manually. In this example, we will request that BioNeMo fetches the sequence for the alpha-humulene Uniprot entry:

![BioNeMo screenshot](images/image17.png)

Figure 17. Protein folding input

Once you select **Generate**, you will be presented with the following screen. It can take some time for the service to return a result. If you are disconnected or navigate away from the page, the same loading screen below will be displayed when you return.

![BioNeMo screenshot](images/image18.png)

Figure 18. Fetching sequence for Uniprot entry

##### Outputs

When a protein is folded, you can download the output with the green download button on the bottom right of the screen. This will download a PDB file containing the resulting structure.

Alternatively, you can choose to visualize the result and work with a web version of MolStar (Mol* (molstar.org)). MolStar has a number of features for visualizing, modifying, and performing measurements on the predicted fold directly. Visualization options are presented by the small vertical toolbar on the right side of the image.

![BioNeMo screenshot](images/image19.png)

Figure 19. Protein folding generation

Clicking on components of the generated molecules will reveal further atomic structure. Selecting multiple components will similarly, generate multiple atomic structures.


#### Docking

##### Inputs

To generate predicted poses from a given ligand and protein, navigate to the DiffDock page. Two files are expected as inputs, the molecule (ligand) and the protein. Molecule files should be submitted either as sdf or mol2 files. Protein files should be uploaded as pdb files. When these are uploaded, there should be green checkmarks next to each of the uploaded files.

![BioNeMo screenshot](images/image20.png)

Figure 20. Docking input parameters

Next is setting the inference parameters which control both the diffusion process and the number of candidate poses to generate. Both parameters control the intensity of the diffusion process. Higher values will result in more accurate predictions, but also increase the computational cost of the algorithms. When both files have uploaded and diffusion parameters are set, select **Generate** to begin the inference process.

##### Outputs

Upon completion, each generated pose will be loaded into the result viewer along with its pose. You can choose to visualize the result and work with the provided web version of MolStar (Mol* (molstar.org)). MolStar has a number of features for visualizing, modifying, and performing measurements on the predicted fold directly. Visualization options are presented by the small vertical toolbar on the right side of the image. Select specific predictions that you would like to visualize.

![BioNeMo screenshot](images/image21.png)

Figure 21. Docking output

Select download to download a zip archive of all the generated poses, the original protein target, and the confidence scores for each. The downloaded archive should look similar to the image below.

![BioNeMo screenshot](images/image22.png)

Figure 22. Downloading zip archive of generated poses