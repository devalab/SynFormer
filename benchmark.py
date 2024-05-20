import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from typing import Optional, Tuple

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


template_codes = [f'<RX_{i+1}>' for i in range(10)]

template_names = [
    'Heteroatom alkylation and arylation',
    'Acylation and related processes',
    'C-C bond formation',
    'Heterocycle formation',
    'Protections',
    'Deprotections',
    'Reductions',
    'Oxidations',
    'Functional group conversions (FGI)',
    'Functional group additions (FGA)'
]


def smi2validmol(smi: str) -> Optional[Chem.Mol]:
    '''
    converts a SMILES string to a valid RDKit molecule
    smi: SMILES string
    returns: RDKit molecule
    '''
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return mol

def concat_molecules(mol_list: list) -> Optional[Chem.Mol]:
    '''
    concatenates a list of molecules into a single molecule
    mol_list: list of molecules
    returns: concatenated molecule
    '''
    try:
        concat_mol = Chem.MolFromSmiles('.'.join([Chem.MolToSmiles(mol) for mol in mol_list]))
        return concat_mol
    except:
        return None
    
def filter_small_mols(mol_list: list[Chem.Mol], min_atoms: int=3) -> list[Chem.Mol]:
    '''
    filters a list of molecules by minumim number of atoms
    mol_list: list of molecules
    min_atoms: minimum number of atoms
    returns: filtered list of molecules
    '''
    filtered_mols = []
    for mol in mol_list:
        if mol.GetNumAtoms() >= min_atoms:
            filtered_mols.append(mol)
    return filtered_mols
    
def compute_accuracy(target: list[str], predicted: list[str], min_atoms: int=-1) -> Tuple[float, float, float, list[str]]:
    '''
    finds the accuracy of a list of predicted SMILES strings
    target: list of target SMILES strings
    predicted: list of predicted SMILES strings
    min_atoms: minimum number of atoms in a molecule
    returns: accuracy, partial accuracy, list of correct SMILES strings
    '''
    target_mols = [smi2validmol(smi) for smi in target]
    predicted_mols = [smi2validmol(smi) for smi in predicted]
    
    # Remove None values
    target_mols = [mol for mol in target_mols if mol is not None]
    predicted_mols = [mol for mol in predicted_mols if mol is not None]

    # filter small molecules
    target_mols = filter_small_mols(target_mols, min_atoms=min_atoms) if min_atoms > 0 else target_mols
    predicted_mols = filter_small_mols(predicted_mols, min_atoms=min_atoms) if min_atoms > 0 else predicted_mols

    # ensure that there are molecules in both lists
    if len(target_mols) == 0 or len(predicted_mols) == 0:
        return 0, 0, 0, []

    interesting_molecules = []
    correct = 0
    adjusted_correct = 0
    for p_mol in predicted_mols:
        for t_mol in target_mols:
            p_smi = Chem.CanonSmiles(Chem.MolToSmiles(p_mol))
            t_smi = Chem.CanonSmiles(Chem.MolToSmiles(t_mol))
            if t_mol.HasSubstructMatch(p_mol) and p_mol.HasSubstructMatch(t_mol):
                if p_smi == t_smi:
                    correct += 1
                else:
                    interesting_molecules.append((t_smi, p_smi))
                adjusted_correct += 1
    accuracy = float(correct == len(target_mols) and len(target_mols) == len(predicted_mols))
    adjusted_accuracy = float(adjusted_correct == len(target_mols) and len(target_mols) == len(predicted_mols))
    partial_accuracy = adjusted_correct/len(target_mols)
    
    return accuracy, adjusted_accuracy, partial_accuracy, interesting_molecules

def halogen_correction(target: list[str], predicted: list[str], min_atoms: int=-1) -> Tuple[float, float, float, list[str]]:
    '''
    replaces all halogens in the target and predicted SMILES strings with iodine and computes the accuracy
    target: list of target SMILES strings
    predicted: list of predicted SMILES strings
    min_atoms: minimum number of atoms in a molecule
    returns: accuracy, partial accuracy, list of correct SMILES strings
    '''
    halogens = ['F', 'Cl', 'Br', 'I', 'At', 'Ts']
    halogen_rep = 'I'

    for halogen in halogens:
        target = [smi.replace(halogen, halogen_rep) for smi in target]
        predicted = [smi.replace(halogen, halogen_rep) for smi in predicted]

    return compute_accuracy(target, predicted, min_atoms=min_atoms)

def compute_tanimoto(target: list[str], predicted: list[str], min_atoms: int=-1) -> float:
    '''
    computes the tanimoto similarity between the target and predicted SMILES strings
    target: list of target SMILES strings
    predicted: list of predicted SMILES strings
    min_atoms: minimum number of atoms in a molecule
    returns: tanimoto similarity
    '''
    target_mols = [smi2validmol(smi) for smi in target]
    predicted_mols = [smi2validmol(smi) for smi in predicted]
    
    # Remove None values
    target_mols = [mol for mol in target_mols if mol is not None]
    predicted_mols = [mol for mol in predicted_mols if mol is not None]

    # filter small molecules
    target_mols = filter_small_mols(target_mols, min_atoms=min_atoms) if min_atoms > 0 else target_mols
    predicted_mols = filter_small_mols(predicted_mols, min_atoms=min_atoms) if min_atoms > 0 else predicted_mols

    # concatenate molecules
    target_mol_concat = concat_molecules(target_mols)
    predicted_mol_concat = concat_molecules(predicted_mols)

    if target_mol_concat is not None and predicted_mol_concat is not None:
        t_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol_concat, 3, nBits=2048)
        p_fp = AllChem.GetMorganFingerprintAsBitVect(predicted_mol_concat, 3, nBits=2048)
        return DataStructs.TanimotoSimilarity(t_fp, p_fp)
    else:
        return 0.0


def get_metrics(target_list: list[list[str]], predicted_list: list[list[str]], 
                apply_halogen_correction: bool=False, min_atoms: int=-1) -> Tuple[list[float], list[float], list[float], list[float], list[str]]:
    '''
    computes the accuracy, partial accuracy, and tanimoto similarity for a list of target and predicted SMILES strings
    target_list: list of target SMILES strings
    predicted_list: list of predicted SMILES strings
    apply_halogen_correction: whether to apply halogen correction
    min_atoms: minimum number of atoms in a molecule
    returns: accuracy, partial accuracy, tanimoto similarity
    '''
    accuracy_list = []
    adjusted_accuracy_list = []
    partial_accuracy_list = []
    tanimoto_list = []
    interesting_molecules = []
    for target, predicted in tqdm(zip(target_list, predicted_list), total=len(target_list), desc=f'Computing metrics with halogen correction {apply_halogen_correction}'):
        accuracy, adjusted_accuracy, partial_accuracy, i_mols = compute_accuracy(target, predicted, min_atoms) if not apply_halogen_correction else halogen_correction(target, predicted, min_atoms)
        accuracy_list.append(accuracy)
        adjusted_accuracy_list.append(adjusted_accuracy)
        partial_accuracy_list.append(partial_accuracy)
        interesting_molecules.extend(i_mols)
        tanimoto_list.append(compute_tanimoto(target, predicted, min_atoms))
    return accuracy_list, adjusted_accuracy_list, partial_accuracy_list, tanimoto_list, interesting_molecules


def compute_score(accuracy_list: list[float], adjusted_accuracy_list: list[float], partial_accuracy_list: list[float], tanimoto_list: list[float], 
                  weights: list[float]) -> float:
    '''
    computes the score for a list of accuracy, partial accuracy, and tanimoto similarity values
    accuracy_list: list of accuracy values
    partial_accuracy_list: list of partial accuracy values
    tanimoto_list: list of tanimoto similarity values
    weights: list of weights for the accuracy, partial accuracy, and tanimoto similarity values
    returns: weighted score
    '''

    weights = np.exp(weights)/np.sum(np.exp(weights))
    assert len(weights) == 4, 'weights must be a list of length 4'
    assert np.isclose(np.sum(weights), 1.0), 'weights must sum to 1'

    score = np.mean([accuracy_list, adjusted_accuracy_list, partial_accuracy_list, tanimoto_list], axis=1) @ weights
    return score


def print_scores(accuracy_list: list[float], adjusted_accuracy_list: list[float], partial_accuracy_list: list[float], tanimoto_list: list[float],
                 accuracy_list_hc: list[float], adjusted_accuracy_list_hc: list[float], partial_accuracy_list_hc: list[float], tanimoto_list_hc: list[float],
                 weights: list[float], table_name: str) -> None:
    '''
    prints the scores for a list of accuracy, partial accuracy, and tanimoto similarity values
    accuracy_list: list of accuracy values
    partial_accuracy_list: list of partial accuracy values
    tanimoto_list: list of tanimoto similarity values
    accuracy_list_hc: list of accuracy values with halogen correction
    partial_accuracy_list_hc: list of partial accuracy values with halogen correction
    tanimoto_list_hc: list of tanimoto similarity values with halogen correction
    weights: list of weights for the accuracy, partial accuracy, and tanimoto similarity values
    table_name: name of the table
    '''

    # stats over normal computation
    acc = np.mean(accuracy_list)
    a_acc = np.mean(adjusted_accuracy_list)
    p_acc = np.mean(partial_accuracy_list)
    tan = np.mean(tanimoto_list)
    
    # stats over halogen replacement
    acc_hc = np.mean(accuracy_list_hc)
    a_acc_hc = np.mean(adjusted_accuracy_list_hc)
    p_acc_hc = np.mean(partial_accuracy_list_hc)
    tan_hc = np.mean(tanimoto_list_hc)
    
    # weighted scores
    score = compute_score(accuracy_list, adjusted_accuracy_list, partial_accuracy_list, tanimoto_list, weights=weights)
    score_hc = compute_score(accuracy_list_hc, adjusted_accuracy_list_hc, partial_accuracy_list_hc, tanimoto_list_hc, weights=weights)

    # print scores using pretty table
    table = PrettyTable()
    table.title = f'Computed Metrics for {table_name}'
    table.field_names = ['Metric', 'Original', 'Halogen Correction', 'Final Index']
    table.add_row(['Accuracy', f'{acc:.3f}', f'{acc_hc:.3f}', f'{(acc_hc+acc)/2:.3f}'])
    table.add_row(['Adjusted Accuracy', f'{a_acc:.3f}', f'{a_acc_hc:.3f}', f'{(a_acc_hc+a_acc)/2:.3f}'])
    table.add_row(['Partial Accuracy', f'{p_acc:.3f}', f'{p_acc_hc:.3f}', f'{(p_acc_hc+p_acc)/2:.3f}'])
    table.add_row(['Adjusted Tanimoto', f'{tan:.3f}', f'{tan_hc:.3f}', f'{(tan_hc+tan)/2:.3f}'])
    table.add_row(['Our Score', f'{score:.3f}', f'{score_hc:.3f}', f'{(score_hc+score)/2:.3f}'])
    print(table)

    return table


class Metrics:
    def __init__(self, target_list: list[list[str]], predicted_list: list[list[str]],  table_name: str, weights: list[float]=[1, 1, 1, 0.5], min_atoms: int=-1):
        '''
        computes the accuracy, partial accuracy, and tanimoto similarity for a list of target and predicted SMILES strings
        target_list: list of target SMILES strings
        predicted_list: list of predicted SMILES strings
        table_name: name of the table
        weights: list of weights for the accuracy, partial accuracy, and tanimoto similarity values
        min_atoms: minimum number of atoms in a molecule
        '''
        self.target_list = target_list
        self.predicted_list = predicted_list
        self.table_name = table_name
        self.weights = weights
        self.min_atoms = min_atoms

        self.accuracy_list, self.adjusted_accuracy_list, self.partial_accuracy_list, self.tanimoto_list, self.interesting_molecules = get_metrics(target_list, predicted_list, apply_halogen_correction=False, min_atoms=min_atoms)
        self.accuracy_list_hc, self.adjusted_accuracy_list_hc, self.partial_accuracy_list_hc, self.tanimoto_list_hc, self.interesting_molecules_hc = get_metrics(target_list, predicted_list, apply_halogen_correction=True, min_atoms=min_atoms)

    def print_metrics(self, weights=None) -> None:
        '''
        prints the scores for a list of accuracy, partial accuracy, and tanimoto similarity values
        weights: list of weights for the accuracy, partial accuracy, and tanimoto similarity values
        '''
        weights = self.weights if weights is None else weights
        t = print_scores(self.accuracy_list, self.adjusted_accuracy_list, self.partial_accuracy_list, self.tanimoto_list,
                         self.accuracy_list_hc, self.adjusted_accuracy_list_hc, self.partial_accuracy_list_hc, self.tanimoto_list_hc,
                         weights, self.table_name)
        return t
        
    def get_metrics(self, weights=None) -> dict[float]:
        '''
        returns the scores for a list of accuracy, partial accuracy, and tanimoto similarity values
        '''
        weights = self.weights if weights is None else weights
        
        # stats over normal computation
        acc = np.mean(self.accuracy_list)
        a_acc = np.mean(self.adjusted_accuracy_list)
        p_acc = np.mean(self.partial_accuracy_list)
        tan = np.mean(self.tanimoto_list)
        
        # stats over halogen replacement
        acc_hc = np.mean(self.accuracy_list_hc)
        a_acc_hc = np.mean(self.adjusted_accuracy_list_hc)
        p_acc_hc = np.mean(self.partial_accuracy_list_hc)
        tan_hc = np.mean(self.tanimoto_list_hc)
        
        # weighted scores
        score = compute_score(self.accuracy_list, self.adjusted_accuracy_list, self.partial_accuracy_list, self.tanimoto_list, weights=weights)
        score_hc = compute_score(self.accuracy_list_hc, self.adjusted_accuracy_list_hc, self.partial_accuracy_list_hc, self.tanimoto_list_hc, weights=weights)
        
        return {
            'accuracy': acc,
            'adjusted_accuracy': a_acc,
            'partial_accuracy': p_acc,
            'tanimoto': tan,
            'score': score,
            'accuracy_hc': acc_hc,
            'adjusted_accuracy_hc': a_acc_hc,
            'partial_accuracy_hc': p_acc_hc,
            'tanimoto_hc': tan_hc,
            'score_hc': score_hc,
            'index': (score+score_hc)/2
        }
