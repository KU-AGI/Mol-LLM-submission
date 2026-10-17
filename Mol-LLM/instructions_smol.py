forward_reaction_prediction = [
    "<INPUT> Based on the reactants and reagents given above, suggest a possible product.",
    "Based on the given reactants and reagents: <INPUT>, what product could potentially be produced?",
    "Given the following reactants and reagents, please provide a possible product. <INPUT>",
    "<INPUT> Given the above reactants and reagents, what could be a probable product of their reaction?",
    "Please provide a feasible product that could be formed using these reactants and reagents: <INPUT> .",
    "Consider that for a chemical reaction, if <INPUT> is/are the reactants and reagents, what can be the product?",
    "Propose a potential product given these reactants and reagents. <INPUT>",
    "Predict the product of a chemical reaction with <INPUT> as the reactants and reagents.",
    "Can you tell me the potential product of a chemical reaction that uses <INPUT> as the reactants and reagents?",
    "Using <INPUT> as the reactants and reagents, tell me the potential product.",
    "Predict a possible product from the listed reactants and reagents. <INPUT>",
    "<INPUT> Considering the given starting materials, what might be the resulting product in a chemical reaction?",
    "A chemical reaction has started with the substance(s) <INPUT> as the reactants and reagents, what could be a probable product?",
]

retrosynthesis = [
    "Based on the given product, provide some plausible reactants that might have been utilized to prepare it. <INPUT>",
    "Can you identify the reactant(s) that might result in the given product <INPUT> ?",
    "Given the following product, please provide possible reactants. <INPUT>",
    "Do retrosynthesis with the product <INPUT> .",
    "<INPUT> Given the product provided, propose some possible reactants that could have been employed in its formation.",
    "To synthesis <INPUT>, what are the possible reactants? Write in the SELFIES representation.",
    "Provide the potential reactants that may be used to produce the product <INPUT> .",
    "What reactants could lead to the production of the following product? <INPUT>",
    "With the given product <INPUT>, suggest some likely reactants that were used in its synthesis.",
    "Identify possible reactants that could have been used to create the specified product. <INPUT>",
    "Could you tell which reactants might have been used to generate the following product? <INPUT>",
    "Suggest possible substances that may have been involved in the synthesis of the presented compound. <INPUT>",
    "Can you list the reactants that might result in the chemical product <INPUT> ?",
]

reagent_prediction = [
    "Based on the given chemical reaction, provide some plausible reagents that might have been utilized to prepare it. <INPUT>",
    "Can you identify the reagents that might result in the given chemical reaction <INPUT> ?",
    "Given the following chemical reaction, please provide possible reagents. <INPUT>",
    "<INPUT> Given the chemical reaction provided, propose some possible reagents that could have been employed in its formation.",
    "With the given chemical reaction <INPUT>, suggest some likely reagents that were used in its synthesis.",
    "Can you provide potential reagents for the following chemical reaction <INPUT> ?",
    "Please suggest some possible reagents that could have been used in the following chemical reaction <INPUT>",
    "<INPUT> Given this chemical reaction, what are some reagents that could have been used?",
    "Can you suggest some reagents that might have been used in the given chemical reaction <INPUT> ?",
    "Given the following chemical reaction, what are some potential reagents that could have been employed? <INPUT>",
]

name_conversion_s2f = [
    "<INPUT> is the SELFIES representation of a molecule. What is its molecular formula?",
    "Convert the SELFIES representation of a molecule <INPUT> into molecular formula.",
    "What is the formula of the molecule <INPUT> ?",
    "Can you give the molecular molecular formula of <INPUT> ?",
    "Please write the molecular formula of the molecule <INPUT> .",
    "Given the SELFIES representation <INPUT>, what would be its molecular formula?",
    "The SELFIES representation <INPUT> represents a specific molecule. Can you reveal its molecular formula?",
    "Considering the SELFIES code <INPUT>, can you determine the corresponding molecular formula?",
    "Can you tell me the molecular formula of <INPUT> ?",
    "I'd like to know the molecular formula of <INPUT> . Can you tell me?",
    "What is the molecular formula for the molecule denoted by <INPUT> ?",
    "What is the molecular formula of <INPUT> ?",
    "Please provide the molecular formula for <INPUT> .",
]

name_conversion_s2i = [
    "<INPUT> is the SELFIES representation of a molecule. What is its IUPAC name?",
    "Convert the SELFIES representation of a molecule <INPUT> into IUPAC name.",
    "What is the IUPAC name of the molecule <INPUT> ?",
    "Can you give the IUPAC name of the molecule <INPUT> ?",
    "Please write the IUPAC name of the molecule <INPUT> .",
    "<INPUT> The above is a SELFIES representation. Write the IUPAC name of the corresponding molecule.",
    "Determine the IUPAC name for the molecule represented by the following SELFIES representation: <INPUT> .",
    "What is the IUPAC name for the molecule whose SELFIES representation is <INPUT> ?",
    "Determine the IUPAC name for the molecule denoted by <INPUT> .",
    "Translate the given SELFIES formula of a molecule <INPUT> into its IUPAC name.",
    "Provide the IUPAC name for the molecule represented as <INPUT> .",
    "Convert the following SELFIES notation <INPUT> into its IUPAC nomenclature.",
    "Turn the given SELFIES symbol of a molecule <INPUT> into its respective IUPAC name.",
]

name_conversion_i2s = [
    "<INPUT> is the IUPAC name of a molecule. Please give its SELFIES representation.",
    "Convert the IUPAC name of a molecule <INPUT> into SELFIES representation.",
    "What is the SELFIES representation of the molecule with IUPAC name <INPUT> ?",
    "Can you give the SELFIES notation of the molecule <INPUT> ?",
    "Please write the SELFIES representation of the molecule <INPUT> .",
    "<INPUT> The above is the IUPAC name of a molecule. Write its SELFIES notation.",
    "The IUPAC name of a certain molecule is <INPUT> . Can you provide its SELFIES representation?",
    "Please identify the SELFIES representation of the molecule named <INPUT> .",
    "For the molecule with <INPUT> as the IUPAC name, what is the corresponding SELFIES representation?",
    "What is the SELFIES notation for <INPUT> ?",
    "Could you provide the SELFIES for <INPUT> ?",
    "Can you tell me the SELFIES representation for the molecule <INPUT> ?",
    "What is the SELFIES representation for <INPUT> ?",
]

name_conversion_i2f = [
    "<INPUT> is the IUPAC name of a molecule. Please give its molecular formula.",
    "Convert the IUPAC name of a molecule <INPUT> into molecular formula.",
    "What is the molecular formula of the molecule <INPUT> ?",
    "Can you give the molecular formula of the molecule <INPUT> ?",
    "Please write the molecular formula of the molecule <INPUT> .",
    "<INPUT> The above is the IUPAC name of a molecule. Write its molecular formula.",
    "What is the molecular formula of the molecule identified by <INPUT> ?",
    "For the molecule named <INPUT> by IUPAC nomenclature, provide its molecular formula.",
    "Name the molecular formula for the molecule established as <INPUT> using the IUPAC nomenclature.",
    "Under IUPAC nomenclature, a molecule named <INPUT> is given. What is its corresponding molecular formula?",
    "Translate the chemical IUPAC name <INPUT> into its molecular formula.",
    "Provide the molecular formula for the IUPAC named substance <INPUT> .",
    "What is the molecular formula of the compound with this IUPAC name <INPUT> ?",
]

molecule_captioning = [
    "Could you give me a brief introduction to this molecule? <INPUT>",
    "Describe this molecule: <INPUT>",
    "Please give me some details about this molecule. <INPUT>",
    "Please provide a brief introduction to this molecule. <INPUT>",
    "Tell me something about this molecule: <INPUT>",
    "<INPUT> The above is a compound. Could you please tell me something about it?",
    "<INPUT> What do you know about the molecule?",
    "Here is a molecule represented with SELFIES: <INPUT> . Please describe it in natural language.",
    "Can you briefly describe the molecular encoded by this SELFIES notation? <INPUT>",
    "I need a brief explanation of the molecule denoted in this SELFIES notation. <INPUT>",
    "I'd like a short overview about this molecule. Can you do that? <INPUT>",
    "May I have a capsulized explanation for this molecule? <INPUT>",
    "What can you tell me about this molecule? <INPUT>",
]

molecule_generation = [
    "Based on the given information, generate a molecule that meets the desired specifications: <INPUT>",
    "Give me a molecule that satisfies the conditions outlined in the description: <INPUT>",
    "Generate a molecule based on this description: <INPUT>",
    "Can you create a molecule that matches the given characteristics? <INPUT>",
    "I need a molecule that meets the following conditions: <INPUT> Please represent the molecule in SELFIES.",
    "Suppose there is a molecule that meets the following description: <INPUT> Please write the SELFIES representation of it.",
    "<INPUT> Use the above information to create a molecule.",
    "Build a molecule that meets the requirement: <INPUT>",
    "Generate a molecule that fulfills the requirement: <INPUT>",
    "Conceptualize a molecule that meets the specified attribute(s): <INPUT>",
    "Come up with a molecule based on the description: <INPUT>",
    "Could you please return a molecule that adheres to this description? <INPUT>",
    "I give you a description of a molecule, and you need to return one molecule in SELFIES that meets the description. The description: <INPUT>",
]

filtering_template_homo = [
    "I would like to know the highest occupied molecular orbital (HOMO) energy of this molecule, could you please provide it?",
    "Please provide the HOMO energy value for this molecule.",
    "I am interested in the HOMO energy of this molecule, could you tell me what it is?",
    "What is the highest occupied molecular orbital (HOMO) energy of this molecule?",
    "Could you give me the HOMO energy value of this molecule?",
    "What is the HOMO energy of this molecule?",
    "Please provide the highest occupied molecular orbital (HOMO) energy value for this molecule.",
    "Please provide me with the HOMO energy value of this molecule.",
    "What is the HOMO level of energy for this molecule?",
    "I would like to know the HOMO energy of this molecule, could you please provide it?",
    "Can you tell me the value of the HOMO energy for this molecule?",
    "Please provide the highest occupied molecular orbital (HOMO) energy of this molecule.",
]

filtering_template_lumo = [
    "Please provide me with the LUMO energy value of this molecule.",
    "I am interested in the LUMO energy of this molecule, could you tell me what it is?",
    "I would like to know the lowest unoccupied molecular orbital (LUMO) energy of this molecule, could you please provide it?",
    "What is the LUMO energy of this molecule?",
    "What is the LUMO level of energy for this molecule?",
    "I would like to know the LUMO energy of this molecule, could you please provide it?",
    "What is the lowest unoccupied molecular orbital (LUMO) energy of this molecule?",
    "Could you give me the LUMO energy value of this molecule?",
    "Please provide the lowest unoccupied molecular orbital (LUMO) energy value for this molecule.",
    "Please provide the lowest unoccupied molecular orbital (LUMO) energy of this molecule.",
    "Can you tell me the value of the LUMO energy for this molecule?",
    "Please provide the LUMO energy value for this molecule.",
]

filtering_template_homo_lumo_gap = [
    "Please provide the gap between HOMO and LUMO of this molecule.",
    "I would like to know the HOMO-LUMO gap of this molecule, can you provide it?",
    "Please give me the HOMO-LUMO gap energy for this molecule.",
    "Can you give me the energy difference between the HOMO and LUMO orbitals of this molecule?",
    "Please provide the energy separation between the highest occupied and lowest unoccupied molecular orbitals (HOMO-LUMO gap) of this molecule.",
    "I need to know the HOMO-LUMO gap energy of this molecule, could you please provide it?",
    "What is the energy separation between the HOMO and LUMO of this molecule?",
    "Could you tell me the energy difference between HOMO and LUMO for this molecule?",
    "What is the HOMO-LUMO gap of this molecule?",
]

qm9_homo = [
    "<INPUT> Based on the molecule given above, suggest the HOMO energy.",
    "Based on the given molecule: <INPUT>, what HOMO energy could potentially be measured?",
    "Given the following molecule, please provide the HOMO energy. <INPUT>",
    "<INPUT> Given the above molecule, what could be the probable HOMO energy?",
    "Please provide the HOMO energy value for this molecule: <INPUT>.",
    "Consider that for this molecule, if <INPUT> is given, what is the HOMO energy?",
    "Propose the HOMO energy value given this molecule. <INPUT>",
    "Predict the HOMO energy of the molecule <INPUT>.",
    "Can you tell me the HOMO energy of the molecule that uses <INPUT>?",
    "Using <INPUT> as the molecule, tell me the HOMO energy value.",
    "Predict the possible HOMO energy for the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what might be the HOMO energy?",
    "A molecule <INPUT> is given; what could be the HOMO energy?",
]

qm9_lumo = [
    "<INPUT> Based on the molecule given above, suggest the LUMO energy.",
    "Based on the given molecule: <INPUT>, what LUMO energy could potentially be measured?",
    "Given the following molecule, please provide the LUMO energy. <INPUT>",
    "<INPUT> Given the above molecule, what could be the probable LUMO energy?",
    "Please provide the LUMO energy value for this molecule: <INPUT>.",
    "Consider that for this molecule, if <INPUT> is given, what is the LUMO energy?",
    "Propose the LUMO energy value given this molecule. <INPUT>",
    "Predict the LUMO energy of the molecule <INPUT>.",
    "Can you tell me the LUMO energy of the molecule that uses <INPUT>?",
    "Using <INPUT> as the molecule, tell me the LUMO energy value.",
    "Predict the possible LUMO energy for the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what might be the LUMO energy?",
    "A molecule <INPUT> is given; what could be the LUMO energy?",
]

qm9_homo_lumo_gap = [
    "<INPUT> Based on the molecule given above, suggest the HOMO-LUMO gap.",
    "Based on the given molecule: <INPUT>, what HOMO-LUMO gap could potentially be measured?",
    "Given the following molecule, please provide the HOMO-LUMO gap. <INPUT>",
    "<INPUT> Given the above molecule, what could be the probable HOMO-LUMO gap?",
    "Please provide the HOMO-LUMO gap value for this molecule: <INPUT>.",
    "Consider that for this molecule, if <INPUT> is given, what is the HOMO-LUMO gap?",
    "Propose the HOMO-LUMO gap value given this molecule. <INPUT>",
    "Predict the HOMO-LUMO gap of the molecule <INPUT>.",
    "Can you tell me the HOMO-LUMO gap of the molecule that uses <INPUT>?",
    "Using <INPUT> as the molecule, tell me the HOMO-LUMO gap value.",
    "Predict the possible HOMO-LUMO gap for the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what might be the HOMO-LUMO gap?",
    "A molecule <INPUT> is given; what could be the HOMO-LUMO gap?",
]

qm9_dipole_moment = [
    "<INPUT> Based on the molecule given above, could you please provide its dipole moment?",
    "Based on the given molecule: <INPUT>, could you please provide its dipole moment?",
    "Given the following molecule, please provide its dipole moment. <INPUT>",
    "<INPUT> Given the above molecule, could you please provide its dipole moment?",
    "Please provide the dipole moment that could be measured for this molecule: <INPUT>.",
    "Consider that for a molecule, if <INPUT> is given, what is its dipole moment?",
    "Propose the dipole moment for this molecule. <INPUT>",
    "Predict the dipole moment of a molecule with <INPUT>.",
    "Can you tell me the dipole moment of a molecule that has the structure <INPUT>?",
    "Using <INPUT> as the molecule, could you please provide its dipole moment?",
    "Predict the dipole moment from the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what is its dipole moment?",
    "A molecule has the structure <INPUT>, what could be its dipole moment?",
]

qm9_isotropic_polarizability = [
    "<INPUT> Based on the molecule given above, could you please provide its isotropic polarizability?",
    "Based on the given molecule: <INPUT>, what is its isotropic polarizability?",
    "Given the following molecule, please provide its isotropic polarizability. <INPUT>",
    "<INPUT> Given the above molecule, what is its isotropic polarizability?",
    "Please provide a feasible isotropic polarizability value for this molecule: <INPUT>.",
    "Consider that for a molecule, if <INPUT> is given, what is its isotropic polarizability?",
    "Propose the isotropic polarizability for this molecule. <INPUT>",
    "Predict the isotropic polarizability of a molecule with <INPUT>.",
    "Can you tell me the isotropic polarizability of a molecule that has the structure <INPUT>?",
    "Using <INPUT> as the molecule, please tell me its isotropic polarizability.",
    "Predict the isotropic polarizability from the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what is its isotropic polarizability?",
    "A molecule has the structure <INPUT>, what could be its isotropic polarizability?",
]

qm9_electronic_spatial_extent = [
    "<INPUT> Based on the molecule given above, could you please provide its electronic spatial extent?",
    "Based on the given molecule: <INPUT>, what is its electronic spatial extent?",
    "Given the following molecule, please provide its electronic spatial extent. <INPUT>",
    "<INPUT> Given the above molecule, what is its electronic spatial extent?",
    "Please provide the electronic spatial extent that could be measured for this molecule: <INPUT>.",
    "Consider that for a molecule, if <INPUT> is given, what is its electronic spatial extent?",
    "Propose the electronic spatial extent for this molecule. <INPUT>",
    "Predict the electronic spatial extent of a molecule with <INPUT>.",
    "Can you tell me the electronic spatial extent of a molecule that has the structure <INPUT>?",
    "Using <INPUT> as the molecule, please tell me its electronic spatial extent.",
    "Predict the electronic spatial extent from the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what is its electronic spatial extent?",
    "A molecule has the structure <INPUT>, what could be its electronic spatial extent?",
]

qm9_zero_point_vibrational_energy = [
    "<INPUT> Based on the molecule given above, could you please provide its zero point vibrational energy?",
    "Based on the given molecule: <INPUT>, what is its zero point vibrational energy?",
    "Given the following molecule, please provide its zero point vibrational energy. <INPUT>",
    "<INPUT> Given the above molecule, what is its zero point vibrational energy?",
    "Please provide a feasible zero point vibrational energy value for this molecule: <INPUT>.",
    "Consider that for a molecule, if <INPUT> is given, what is its zero point vibrational energy?",
    "Propose the zero point vibrational energy for this molecule. <INPUT>",
    "Predict the zero point vibrational energy of a molecule with <INPUT>.",
    "Can you tell me the zero point vibrational energy of a molecule that has the structure <INPUT>?",
    "Using <INPUT> as the molecule, please tell me its zero point vibrational energy.",
    "Predict the zero point vibrational energy from the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what is its zero point vibrational energy?",
    "A molecule has the structure <INPUT>, what could be its zero point vibrational energy?",
]

qm9_heat_capacity_298K = [
    "<INPUT> Based on the molecule given above, could you please provide its heat capacity at 298.15K?",
    "Based on the given molecule: <INPUT>, what is its heat capacity at 298.15K?",
    "Given the following molecule, please provide its heat capacity at 298.15K. <INPUT>",
    "<INPUT> Given the above molecule, what could be its heat capacity at 298.15K?",
    "Please provide a feasible heat capacity at 298.15K that could be measured for this molecule: <INPUT>.",
    "Consider that for a molecule, if <INPUT> is given, what can be its heat capacity at 298.15K?",
    "Propose the heat capacity at 298.15K for this molecule. <INPUT>",
    "Predict the heat capacity at 298.15K of a molecule with <INPUT>.",
    "Can you tell me the heat capacity at 298.15K of a molecule that has the structure <INPUT>?",
    "Using <INPUT> as the molecule, please tell me its heat capacity at 298.15K.",
    "Predict the heat capacity at 298.15K from the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what might be its heat capacity at 298.15K?",
    "A molecule has the structure <INPUT>, what could be its heat capacity at 298.15K?",
]

qm9_internal_energy_298K = [
    "<INPUT> Based on the molecule given above, could you please provide its internal energy at 298.15K?",
    "Based on the given molecule: <INPUT>, what is its internal energy at 298.15K?",
    "Given the following molecule, please provide its internal energy at 298.15K. <INPUT>",
    "<INPUT> Given the above molecule, what could be its internal energy at 298.15K?",
    "Please provide a feasible internal energy at 298.15K that could be measured for this molecule: <INPUT>.",
    "Consider that for a molecule, if <INPUT> is given, what can be its internal energy at 298.15K?",
    "Propose the internal energy at 298.15K for this molecule. <INPUT>",
    "Predict the internal energy at 298.15K of a molecule with <INPUT>.",
    "Can you tell me the internal energy at 298.15K of a molecule that has the structure <INPUT>?",
    "Using <INPUT> as the molecule, please tell me its internal energy at 298.15K.",
    "Predict the internal energy at 298.15K from the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what might be its internal energy at 298.15K?",
    "A molecule has the structure <INPUT>, what could be its internal energy at 298.15K?",
]

qm9_enthalpy_298K = [
    "<INPUT> Based on the molecule given above, could you please provide its enthalpy at 298.15K?",
    "Based on the given molecule: <INPUT>, what is its enthalpy at 298.15K?",
    "Given the following molecule, please provide its enthalpy at 298.15K. <INPUT>",
    "<INPUT> Given the above molecule, what could be its enthalpy at 298.15K?",
    "Please provide a feasible enthalpy at 298.15K that could be measured for this molecule: <INPUT>.",
    "Consider that for a molecule, if <INPUT> is given, what can be its enthalpy at 298.15K?",
    "Propose the enthalpy at 298.15K for this molecule. <INPUT>",
    "Predict the enthalpy at 298.15K of a molecule with <INPUT>.",
    "Can you tell me the enthalpy at 298.15K of a molecule that has the structure <INPUT>?",
    "Using <INPUT> as the molecule, please tell me its enthalpy at 298.15K.",
    "Predict the enthalpy at 298.15K from the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what might be its enthalpy at 298.15K?",
    "A molecule has the structure <INPUT>, what could be its enthalpy at 298.15K?",
]

qm9_free_energy_298K = [
    "<INPUT> Based on the molecule given above, could you please provide its free energy at 298.15K?",
    "Based on the given molecule: <INPUT>, what is its free energy at 298.15K?",
    "Given the following molecule, please provide its free energy at 298.15K. <INPUT>",
    "<INPUT> Given the above molecule, what could be its free energy at 298.15K?",
    "Please provide a feasible free energy at 298.15K that could be measured for this molecule: <INPUT>.",
    "Consider that for a molecule, if <INPUT> is given, what can be its free energy at 298.15K?",
    "Propose the free energy at 298.15K for this molecule. <INPUT>",
    "Predict the free energy at 298.15K of a molecule with <INPUT>.",
    "Can you tell me the free energy at 298.15K of a molecule that has the structure <INPUT>?",
    "Using <INPUT> as the molecule, please tell me its free energy at 298.15K.",
    "Predict the free energy at 298.15K from the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what might be its free energy at 298.15K?",
    "A molecule has the structure <INPUT>, what could be its free energy at 298.15K?",
]


bace = [
    "<INPUT> Based on the molecule given above, suggest the biological activity against BACE-1.",
    "Based on the given molecule: <INPUT>, what biological activity could potentially be observed against BACE-1?",
    "Given the following molecule, please provide the biological activity against BACE-1. <INPUT>",
    "<INPUT> Given the above molecule, what could be the probable biological activity against BACE-1?",
    "Please provide the biological activity value for this molecule against BACE-1: <INPUT>.",
    "Consider that for this molecule, if <INPUT> is given, what is the biological activity against BACE-1?",
    "Propose the biological activity value given this molecule against BACE-1. <INPUT>",
    "Predict the biological activity of the molecule <INPUT> against BACE-1.",
    "Can you tell me the biological activity of the molecule that uses <INPUT> against BACE-1?",
    "Using <INPUT> as the molecule, tell me the biological activity against BACE-1.",
    "Predict the possible biological activity against BACE-1 for the listed molecule. <INPUT>",
    "<INPUT> Considering the given molecule, what might be the biological activity against BACE-1?",
    "A molecule <INPUT> is given; what could be the biological activity against BACE-1?",
]

property_prediction_bbbp = [
    "Can <INPUT> pass through the blood-brain barrier (BBB)?",
    "Can <INPUT> permeate the blood-brain barrier?",
    "Does <INPUT> have the blood-brain barrier permeability (BBBP)?",
    "Does <INPUT> have the ability to penetrate the blood-brain barrier?",
    "Is <INPUT> capable of crossing the blood-brain barrier (BBB)?",
    "Can <INPUT> penetrate the blood-brain barrier?",
    "Is <INPUT> capable of crossing the blood-brain barrier (BBB)?",
    "Is it likely for <INPUT> to effectively traverse the blood-brain barrier?",
    "Is blood-brain barrier permeability (BBBP) present in <INPUT> ?",
    "Does <INPUT> demonstrate the potential for blood-brain barrier permeation (BBBP)?",
    "Does blood-brain barrier permeability (BBBP) apply to <INPUT> ?",
    "Is blood-brain barrier permeability (BBBP) a property of <INPUT> ?",
    "Would <INPUT> be able to successfully permeate the blood-brain barrier?",
]

property_prediction_clintox = [
    "Is <INPUT> toxic?",
    "Is the molecule <INPUT> known to exhibit toxic properties to human?",
    "Is <INPUT> classified as a hazardous substance with documented toxicity to human?",
    "Does <INPUT> have documented toxicity to human?",
    "Is <INPUT> considered to be a toxic substance to human?",
    "Does <INPUT> have toxicity to human?",
    "Do you think that <INPUT> is toxic to human?",
    "Can <INPUT> be considered toxic to human?",
    "Is <INPUT> associated with harmful and toxic effects to human?",
    "Is the substance <INPUT> toxic for human?",
    "<INPUT> Would you say that the molecule given above is toxic to human?",
    "<INPUT> Is it toxic to human?",
    "Tell me if <INPUT> is toxic to human.",
]

property_prediction_sider = [
    "Can <INPUT> cause hepatobiliary disorders?",
    "Are hepatobiliary disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause hepatobiliary disorders?",
    "Can <INPUT> cause side effects that affect the hepatobiliary system?",
    "Are there any known side effects of <INPUT> affecting the hepatobiliary system?",
    "Does <INPUT> adversely affect the hepatobiliary system?",
    "Do the side effects of <INPUT> impact the hepatobiliary system?",
    "Are hepatobiliary system functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause metabolism and nutrition disorders?",
    "Are metabolism and nutrition disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause metabolism and nutrition disorders?",
    "Can <INPUT> cause side effects that affect metabolism and nutrition?",
    "Are there any known side effects of <INPUT> affecting metabolism and nutrition?",
    "Does <INPUT> adversely affect metabolism and nutrition?",
    "Do the side effects of <INPUT> impact metabolism and nutrition?",
    "Are metabolism and nutrition functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause eye disorders?",
    "Are eye disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause eye disorders?",
    "Can <INPUT> cause side effects that affect the eyes?",
    "Are there any known side effects of <INPUT> affecting the eyes?",
    "Does <INPUT> adversely affect the eyes?",
    "Do the side effects of <INPUT> impact the eyes?",
    "Are eye functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause musculoskeletal and connective tissue disorders?",
    "Are musculoskeletal and connective tissue disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause musculoskeletal and connective tissue disorders?",
    "Can <INPUT> cause side effects that affect the musculoskeletal system and the connective tissues?",
    "Are there any known side effects of <INPUT> affecting the musculoskeletal system and the connective tissues?",
    "Does <INPUT> adversely affect the musculoskeletal system and the connective tissues?",
    "Do the side effects of <INPUT> impact the musculoskeletal system and the connective tissues?",
    "Are musculoskeletal system and connective tissues functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause gastrointestinal disorders?",
    "Are gastrointestinal disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause gastrointestinal disorders?",
    "Can <INPUT> cause side effects that affect the gastrointestinal system?",
    "Are there any known side effects of <INPUT> affecting the gastrointestinal system?",
    "Does <INPUT> adversely affect the gastrointestinal system?",
    "Do the side effects of <INPUT> impact the gastrointestinal system?",
    "Are gastrointestinal system functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause immune system disorders?",
    "Are immune system disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause immune system disorders?",
    "Can <INPUT> cause side effects that affect the immune system?",
    "Are there any known side effects of <INPUT> affecting the immune system?",
    "Does <INPUT> adversely affect the immune system?",
    "Do the side effects of <INPUT> impact the immune system?",
    "Are immune system functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause reproductive system and breast disorders?",
    "Are reproductive system and breast disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause reproductive system and breast disorders?",
    "Can <INPUT> cause side effects that affect the reproductive system and the breasts?",
    "Are there any known side effects of <INPUT> affecting the reproductive system and the breasts?",
    "Does <INPUT> adversely affect the reproductive system and the breasts?",
    "Do the side effects of <INPUT> impact the reproductive system and the breasts?",
    "Are reproductive system and breast functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause neoplasms?",
    "Are neoplasms a side effect of <INPUT> ?",
    "Is <INPUT> known to cause neoplasms?",
    "Is <INPUT> linked to the development of neoplasms?",
    "Can neoplasms be induced by <INPUT> ?",
    "Does <INPUT> have the potential to trigger neoplasms?",
    "Is there a correlation between <INPUT> and the occurrence of neoplasms?",
    "Could <INPUT> lead to the formation of neoplasms?",
    "Can <INPUT> cause endocrine disorders?",
    "Are endocrine disorders a side effect of <INPUT> ?",
    "Can <INPUT> cause side effects that affect the endocrine system?",
    "Does <INPUT> adversely affect the endocrine system?",
    "Does <INPUT> adversely affect hormones?",
    "Do the side effects of <INPUT> impact the endocrine system?",
    "Do the side effects of <INPUT> impact hormones?",
    "Are endocrine system functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause vascular disorders?",
    "Are vascular disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause vascular disorders?",
    "Can <INPUT> cause side effects that affect the blood vessels?",
    "Are there any known side effects of <INPUT> affecting the blood vessels?",
    "Does <INPUT> adversely affect the blood vessels?",
    "Do the side effects of <INPUT> impact the blood vessels?",
    "Are vascular functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause blood and lymphatic system disorders?",
    "Are blood and lymphatic system disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause blood and lymphatic system disorders?",
    "Can <INPUT> cause side effects that affect the blood and lymphatic system?",
    "Are there any known side effects of <INPUT> affecting the blood and lymphatic system?",
    "Does <INPUT> adversely affect the blood and lymphatic system?",
    "Do the side effects of <INPUT> impact the blood and lymphatic system?",
    "Are blood and lymphatic system functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause skin and subcutaneous tissue disorders?",
    "Are skin and subcutaneous tissue disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause skin and subcutaneous tissue disorders?",
    "Can <INPUT> cause side effects that affect the skin and subcutaneous tissues?",
    "Are there any known side effects of <INPUT> affecting the skin and subcutaneous tissues?",
    "Does <INPUT> adversely affect the skin and subcutaneous tissues?",
    "Do the side effects of <INPUT> impact the skin and subcutaneous tissues?",
    "Are skin and subcutaneous tissues functions adversely affected by <INPUT> ?",
    "Can <INPUT> increase risk of congenital, familial, and genetic disorders?",
    "Is increased risk of congenital, familial, and genetic disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to increase risk of congenital, familial, and genetic disorders?",
    "Does <INPUT> increase risk of congenital, familial, and genetic disorders?",
    "Do the side effects of <INPUT> impact congenital, familial, and genetic disorders?",
    "Can <INPUT> result in an elevated chance of congenital, familial, and genetic disorders?",
    "Can the use of <INPUT> lead to a higher likelihood of congenital, familial, and genetic disorders?",
    "Does <INPUT> contribute to an amplified hazard of congenital, familial, anGastrointestinald genetic disorders?",
    "Can <INPUT> cause respiratory, thoracic, and mediastinal disorders?",
    "Are respiratory, thoracic, and mediastinal disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause respiratory, thoracic, and mediastinal disorders?",
    "Can <INPUT> cause side effects that affect the respiratory system, thorax, and mediastina?",
    "Are there any known side effects of <INPUT> affecting the respiratory system, thorax, and mediastina?",
    "Does <INPUT> adversely affect the respiratory system, thorax, and mediastina?",
    "Do the side effects of <INPUT> impact the respiratory system, thorax, and mediastina?",
    "Are respiratory system, thorax, and mediastina functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause psychiatric disorders?",
    "Are psychiatric disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause psychiatric disorders?",
    "Can <INPUT> cause side effects that affect mental health?",
    "Are there any known side effects of <INPUT> affecting mental health?",
    "Does <INPUT> adversely affect mental health?",
    "Do the side effects of <INPUT> impact mental health?",
    "Is mental health adversely affected by <INPUT> ?",
    "Can <INPUT> cause renal and urinary disorders?",
    "Are renal and urinary disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause renal and urinary disorders?",
    "Can <INPUT> cause side effects that affect the kidneys, bladder, and urinary system?",
    "Are there any known side effects of <INPUT> affecting the kidneys, bladder, and urinary system?",
    "Does <INPUT> adversely affect the kidneys, bladder and urinary system?",
    "Do the side effects of <INPUT> impact the kidneys, bladder, and urinary system?",
    "Are kidneys, bladder, and urinary system functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause issues for pregnant or postpartum women?",
    "Is <INPUT> known to cause issues for pregnant or postpartum women?",
    "Can <INPUT> cause side effects that affect pregnant or postpartum women?",
    "Are there any known side effects of <INPUT> affecting pregnant or postpartum women?",
    "Does <INPUT> have side effects to women's pregnancy, puerperium, and perinatal conditions?",
    "Are there adverse effects of <INPUT> on women's pregnancy, postpartum period, and perinatal health?",
    "Does the use of <INPUT> have any negative impacts on pregnancy, postpartum recovery, or perinatal conditions in women?",
    "Can the consumption of <INPUT> lead to side effects for women in terms of pregnancy, puerperium, and perinatal conditions?",
    "Can <INPUT> cause ear disorders?",
    "Are ear disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause ear disorders?",
    "Can <INPUT> cause side effects that affect the ears and auditory system?",
    "Are there any known side effects of <INPUT> affecting the ears and auditory system?",
    "Does <INPUT> adversely affect the ears and auditory system?",
    "Do the side effects of <INPUT> impact the ears and auditory system?",
    "Are ears and auditory system functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause cardiac disorders?",
    "Are cardiac disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause cardiac disorders?",
    "Can <INPUT> cause side effects that affect the heart?",
    "Are there any known side effects of <INPUT> affecting the heart?",
    "Does <INPUT> adversely affect the heart?",
    "Do the side effects of <INPUT> impact the heart?",
    "Are heart functions adversely affected by <INPUT> ?",
    "Can <INPUT> cause nervous system disorders?",
    "Are nervous system disorders a side effect of <INPUT> ?",
    "Is <INPUT> known to cause nervous system disorders?",
    "Can <INPUT> cause side effects that affect the nervous system?",
    "Are there any known side effects of <INPUT> affecting the nervous system?",
    "Does <INPUT> adversely affect the nervous system?",
    "Do the side effects of <INPUT> impact the nervous system?",
    "Are nervous system functions adversely affected by <INPUT> ?",
]

toxcast = [
    "<INPUT> Based on the molecule given above, suggest if it is toxic in the ACEA_T47D_80hr_Negative assay.",
    "Based on the given molecule: <INPUT>, what is the toxicity status according to the ACEA_T47D_80hr_Negative assay?",
    "Given the following molecule, please provide its toxicity information from the ACEA_T47D_80hr_Negative assay. <INPUT>",
    "<INPUT> Given the above molecule, what could be the probable toxicity result in the ACEA_T47D_80hr_Negative assay?",
    "Please provide the toxicity status for this molecule based on the ACEA_T47D_80hr_Negative assay: <INPUT>.",
    "Consider that for this molecule, if <INPUT> is given, what is its toxicity status according to the ACEA_T47D_80hr_Negative assay?",
    "Propose the toxicity status given this molecule against the ACEA_T47D_80hr_Negative assay. <INPUT>",
    "Predict the toxicity of the molecule <INPUT> in the ACEA_T47D_80hr_Negative assay.",
    "Can you tell me the toxicity status of the molecule that uses <INPUT> in the ACEA_T47D_80hr_Negative assay?",
    "Using <INPUT> as the molecule, tell me the toxicity status in the ACEA_T47D_80hr_Negative assay.",
    "Predict the possible toxicity for the listed molecule in the ACEA_T47D_80hr_Negative assay. <INPUT>",
    "<INPUT> Considering the given molecule, what might be the toxicity result in the ACEA_T47D_80hr_Negative assay?",
    "A molecule <INPUT> is given; what could be the toxicity status in the ACEA_T47D_80hr_Negative assay?",
]

tox21 = [
    "For the molecule <INPUT>, assess its potential toxicity through interactions or inhibitions.",
    "With the molecule <INPUT> provided, determine its toxicity concerning interactions with other substances.",
    "Considering the molecule <INPUT>, what is the likely toxicity from its inhibitory interactions?",
    "Evaluate the toxicity of this molecule <INPUT> based on its effects on other molecules.",
    "If the molecule <INPUT> is given, what is its toxicity due to interactions with other compounds?",
    "Assess the toxicity for this molecule <INPUT> given its interactions with others.",
    "Estimate the toxicity of molecule <INPUT> considering its interactions or inhibitory effects.",
    "What is the toxicity of the molecule <INPUT> caused by its interactions or inhibitions?",
    "Determine the toxicity potential for this molecule <INPUT>, considering its interactions.",
    "Given the molecule <INPUT>, what associated toxicity might arise from its interactions or inhibitions?",
    "For the molecule <INPUT>, what toxicity status could be induced from its interactions or inhibitions?",
]

property_prediction_hiv = [
    "Is <INPUT> known to inhibit HIV replication?",
    "Does <INPUT> inhibit viral replication for HIV?",
    "Could <INPUT> be used to prevent HIV replication?",
    "Can <INPUT> inhibit the replication of human immunodeficiency virus (HIV)?",
    "Could HIV replication be slowed or stopped by <INPUT> ?",
    "Does <INPUT> have an inhibitory impact on HIV?",
    "Can <INPUT> effectively inhibit HIV replication?",
    "Does <INPUT> exhibit inhibitory effects on HIV replication?",
    "Is <INPUT> capable of suppressing HIV replication?",
    "Would <INPUT> have the ability to hinder HIV replication?",
    "Can <INPUT> serve as an inhibitor of HIV replication?",
    "Do you suggest that <INPUT> can impede the replication of HIV?",
    "<INPUT> Predict if the molecule given above have an inhibitory impact on HIV.",
]

property_prediction_esol = [
    "What is the log solubility of <INPUT> in water?",
    "Please predict the log solubility of <INPUT> in water.",
    "Tell me the log solubility of <INPUT> in water.",
    "Could you provide the log solubility of <INPUT> in aqueous solutions?",
    "Give me the log water solubility for the following molecule: <INPUT>.",
    "<INPUT> What is the log solubility in water for the molecule given above?",
    "What is the logarithmic value of the solubility of <INPUT> in water?",
    "Can you predict the water log solubility of <INPUT>?",
    "<INPUT> Could you tell me the log solubility in water for the molecule given above?",
]

# this version is used in v7.1 dataset
property_prediction_esol_used_later = [
    "What is the log solubility of <INPUT> in water?",
    "Please predict the log solubility of <INPUT> in water.",
    "Tell me the log solubility of <INPUT> in water.",
    "Predict the log solubility of <INPUT> in water.",
    "<INPUT> What is the log solubility in water for the molecule given above?",
    "For this molecule <INPUT>, predict the log solubility in water.",
    "Could you provide information on the log solubility of <INPUT> in water?",
    "What is the log solubility of <INPUT> in water?",
    "<INPUT> What is the log solubility in water for the given molecule?",
    "What value does the log solubility in water have for <INPUT>?",
    "What is the log solubility in water associated with <INPUT>?",
    "Provide the log solubility in water of <INPUT>.",
    "<INPUT> Please predict the log solubility in water for the molecule given above.",
]

property_prediction_lipo = [
    "What is the octanol/water distribution coefficient (logD at pH 7.4) of <INPUT> ?",
    "Please predict the octanol/water distribution coefficient (logD at pH 7.4) of <INPUT> .",
    "Tell me the octanol/water distribution coefficient (logD at pH 7.4) of <INPUT> .",
    "Predict the octanol/water distribution coefficient logD under the circumstance of pH 7.4 for <INPUT> .",
    "<INPUT> What is the octanol/water distribution coefficient logD under the circumstance of pH 7.4 for the molecule given above?",
    "For this molecule <INPUT>, predict the octanol/water distribution coefficient (logD at pH 7.4).",
    "Could you provide information on the octanol/water distribution coefficient (logD) of <INPUT> under pH 7.4?",
    "At pH 7.4, what is the logD value (the octanol/water distribution coefficient) of <INPUT> ?",
    "<INPUT> What is the logD (the octanol/water distribution coefficient) at pH 7.4 for the given molecule?",
    "What value does the octanol/water distribution coefficient (logD at pH 7.4) have for <INPUT> ?",
    "What is the logD at pH 7.4 (the octanol/water distribution coefficient) associated with <INPUT> ?",
    "Provide the logD (the octanol/water distribution coefficient) at pH 7.4 of <INPUT> .",
    "<INPUT> Please predict the octanol/water distribution coefficient (logD at pH 7.4) for the molecule given above.",
]


TDC_TEMPLATES = {
    "ppbr_az": [
        "What is the human plasma protein binding rate of <INPUT>, as a percentage?",
        "Please predict what percentage of <INPUT> is bound to plasma proteins in humans.",
        "Tell me the plasma protein binding rate of <INPUT>, in percent.",
        "Could you provide the human plasma protein binding percentage for <INPUT>?",
        "Give me the percentage of the following molecule that binds plasma protein: <INPUT>.",
        "<INPUT> What is the plasma protein binding rate of the molecule given above?",
        "Predict how strongly <INPUT> binds human plasma proteins, as a percentage.",
        "Can you estimate the percentage of <INPUT> that is bound to plasma proteins?",
        "<INPUT> Based on the molecule given above, suggest its human plasma protein binding rate.",
        "For the molecule <INPUT>, what plasma protein binding percentage would you expect?",
        "Propose a human plasma protein binding rate, as a percentage, for this molecule. <INPUT>",
        "A molecule <INPUT> is given; what could its plasma protein binding rate be?",
        "Using <INPUT> as the molecule, tell me its human plasma protein binding rate in percent.",
    ],
    "clearance_hepatocyte_az": [
        "What is the base-10 logarithm of the hepatocyte clearance, in uL/min per million cells, of <INPUT>?",
        "Please predict the base-10 logarithm of the drug clearance of <INPUT> in a human hepatocyte assay.",
        "Tell me log10 of the hepatocyte clearance, in uL/min per million cells, measured for <INPUT>.",
        "Could you provide the base-10 logarithm of the hepatocyte clearance in uL/min per million cells for <INPUT>?",
        "Give me the base-10 logarithm of the human hepatocyte clearance for the following molecule: <INPUT>.",
        "<INPUT> What is the base-10 logarithm of the hepatocyte clearance of the molecule given above?",
        "Predict how fast hepatocytes clear <INPUT>, as the base-10 logarithm of its clearance.",
        "Can you estimate the base-10 logarithm of the hepatocyte clearance of <INPUT>?",
        "<INPUT> Based on the molecule given above, suggest the base-10 logarithm of its hepatocyte clearance.",
        "For the molecule <INPUT>, what base-10 logarithm of hepatocyte clearance in uL/min per million cells would you expect?",
        "Propose a value for the base-10 logarithm of the hepatocyte clearance of this molecule. <INPUT>",
        "A molecule <INPUT> is given; what could the base-10 logarithm of its hepatocyte clearance be?",
        "Using <INPUT> as the molecule, tell me the base-10 logarithm of its clearance in a human hepatocyte assay.",
    ],
}

TDC_SPECIES_NAMES = {"Homo sapiens": "human", "Rattus norvegicus": "rat", "Canis lupus familiaris": "dog", "Mus musculus": "mouse", "Cavia porcellus": "guinea pig"}

_TDC_REVISION_PPBR = (
    "Predict plasma protein binding (%) of <INPUT> in {species} plasma.",
    "What percentage of <INPUT> binds proteins in {species} plasma?",
    "Give the plasma protein binding (%) of <INPUT> in {species} plasma.",
    "Estimate plasma protein binding (%) for <INPUT> in {species} plasma.",
    "Report protein binding in {species} plasma (%) for <INPUT>.",
    "<INPUT> What is its protein binding in {species} plasma (%)?",
    "Predict the percentage of <INPUT> bound to proteins in {species} plasma.",
    "Estimate the protein-bound percentage of <INPUT> in {species} plasma.",
    "<INPUT> Give its plasma protein binding (%) in {species} plasma.",
    "For <INPUT>, predict protein binding in {species} plasma (%).",
    "Provide the plasma protein binding percentage in {species} plasma for <INPUT>.",
    "What is the protein-bound fraction of <INPUT> in {species} plasma, as a percentage?",
    "For the molecule <INPUT>, report protein binding (%) in {species} plasma.",
)

_TDC_REVISION_HEP = (
    "Predict log10 clearance for <INPUT> in {species} hepatocytes",
    "Estimate log10 clearance of <INPUT> in {species} hepatocytes",
    "Report log10 clearance for <INPUT> in {species} hepatocytes",
    "What is log10 clearance of <INPUT> in {species} hepatocytes",
    "Give log10 clearance for <INPUT> in {species} hepatocytes",
    "<INPUT> Predict its log10 clearance in {species} hepatocytes",
    "Predict log10 hepatocyte clearance in {species} for <INPUT>",
    "Estimate log10 hepatocyte clearance in {species} for <INPUT>",
    "<INPUT> Report its log10 clearance in {species} hepatocytes",
    "For <INPUT>, give log10 clearance in {species} hepatocytes",
    "Provide log10 clearance for <INPUT> in {species} hepatocytes",
    "What log10 clearance is expected for <INPUT> in {species} hepatocytes",
    "For the molecule <INPUT>, report log10 clearance in {species} hepatocytes",
)

_TDC_REVISION_HEP_MEASUREMENT = " at 37 C (uL/min/10^6 cells; clip to 3-150 before log10)."

def get_tdc_revision_templates(task: str, species: str) -> tuple[str, ...]:
    if species not in TDC_SPECIES_NAMES:
        raise ValueError(f"Unsupported species: {species}")
    if task == "tdc-ppbr_az":
        return tuple(x.format(species=TDC_SPECIES_NAMES[species]) for x in _TDC_REVISION_PPBR)
    if task == "tdc-clearance_hepatocyte_az":
        return tuple((x + _TDC_REVISION_HEP_MEASUREMENT).format(species=TDC_SPECIES_NAMES[species]) for x in _TDC_REVISION_HEP)
    raise ValueError(f"Unsupported revision task: {task}")


import re
from functools import lru_cache
LEGACY_SYSTEM = 'You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation.'
SPECIES = TDC_SPECIES_NAMES
species_templates = get_tdc_revision_templates
TARGET_TASKS = {'tdc-ppbr_az', 'tdc-clearance_hepatocyte_az'}
REVISION = 'tdc-species-nosystem-v1'

def remove_legacy_system(prompt: str) -> tuple[str, int]:
    match = re.match('((?:<s>)?\\[INST\\][ \\t]*)', prompt)
    if not match:
        return (prompt, 0)
    start = match.end()
    rest = prompt[start:]
    count = 0
    prefix = re.compile(re.escape(LEGACY_SYSTEM) + '[ \\t]*\\r?\\n[ \\t]*\\r?\\n[ \\t]*')
    while True:
        hit = prefix.match(rest)
        if not hit:
            break
        rest = rest[hit.end():]
        count += 1
    if rest.startswith(LEGACY_SYSTEM):
        raise ValueError('Known system boilerplate has an unrecognized separator')
    return (prompt[:start] + rest, count)

def chat_messages(user_prompt: str, mode: str='none') -> list[dict[str, str]]:
    if mode not in {'none', 'legacy'}:
        raise ValueError(f'Unknown system_prompt_mode: {mode}')
    messages = [] if mode == 'none' else [{'role': 'system', 'content': LEGACY_SYSTEM}]
    return messages + [{'role': 'user', 'content': user_prompt}]


def validate_runtime_revision(revision: str | None, eval_molpo: bool, endpoint_matched_donors: bool=False) -> None:
    if revision == REVISION and eval_molpo and (not endpoint_matched_donors):
        raise ValueError('Species-aware GDR/MolPO evaluation needs rebuilt same-condition donors; this native prompt revision only supports eval_molpo=false')

@lru_cache(maxsize=2)
def _endpoint_templates(task):
    compile_template = lambda text: re.compile(re.escape(text).replace(re.escape('<INPUT>'), '.+'), re.S)
    revised = [(species, compile_template(template)) for species in SPECIES for template in species_templates(task, species)]
    legacy = [compile_template(template) for template in TDC_TEMPLATES[task[4:]]]
    return (revised, legacy)

def tdc_prompt_endpoint(row):
    task = row['task']
    if task not in TARGET_TASKS:
        return task
    clean, _ = remove_legacy_system(row.get('prompt_text', ''))
    wrapped = re.fullmatch('(?:<s>)?\\[INST\\] (.*) \\[/INST\\] ', clean, re.S)
    query = wrapped[1] if wrapped else clean
    revised, legacy = _endpoint_templates(task)
    matches = {species for species, pattern in revised if pattern.fullmatch(query)}
    if len(matches) == 1:
        species = next(iter(matches))
        measurement = 'plasma_binding_percent' if task == 'tdc-ppbr_az' else 'hepatocyte_37C_uL_min_1e6cells_clip3_150_log10'
        return f'{task}/{REVISION}/{species}/{measurement}'
    if not matches and any((pattern.fullmatch(query) for pattern in legacy)):
        return task
    raise ValueError(f'{task}: unrecognized/ambiguous species-measurement prompt; register its full template')
