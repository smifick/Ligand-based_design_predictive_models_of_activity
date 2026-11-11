import sys
import pandas as pd
from rdkit.Chem import MACCSkeys
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Avalon import pyAvalonTools
from jpype import isJVMStarted, startJVM, getDefaultJVMPath, JPackage
try:
    from openbabel import pybel
except ModuleNotFoundError:
    import pybel
import numpy as np
from tqdm import tqdm
import misc.longlist as longlist
import os

# working_dir = os.getcwd()


def rdk_fingerprint(smi, fp_type="rdkit", size=2048):
    _fingerprinters = {
        "rdkit": Chem.rdmolops.RDKFingerprint,
        "maccs": MACCSkeys.GenMACCSKeys,
        "TopologicalTorsion": Torsions.GetTopologicalTorsionFingerprint,
        "Avalon": pyAvalonTools.GetAvalonFP
    }
    mol = Chem.MolFromSmiles(smi)
    if fp_type in _fingerprinters:
        fingerprinter = _fingerprinters[fp_type]
        fp = fingerprinter(mol)
    elif fp_type == "AtomPair":
        fp = Pairs.GetAtomPairFingerprintAsBitVect(mol)
    elif fp_type == "Morgan":
        fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=size)
    else:
        raise IOError('invalid fingerprint type')
    if fp_type == "AtomPair":
        res = np.array(fp)
    else:
        temp = fp.GetOnBits()
        res = [i for i in temp]
    return res


def cdk_parser_smiles(cdk, smi):
    sp = cdk.smiles.SmilesParser(cdk.DefaultChemObjectBuilder.getInstance())
    try:
        mol = sp.parseSmiles(smi)
    except Exception:
        raise IOError('invalid smiles input')
    return mol


def cdk_fingerprint(cdk, smi, fp_type="standard", size=2048, depth=6):
    if fp_type == 'maccs':
        nbit = 166
    elif fp_type == 'estate':
        nbit = 79
    elif fp_type == 'pubchem':
        nbit = 881
    elif fp_type == 'klekota-roth':
        nbit = 4860
    else:
        nbit = size

    _fingerprinters = {
        "daylight": cdk.fingerprint.Fingerprinter(size, depth),
        "extended": cdk.fingerprint.ExtendedFingerprinter(size, depth),
        "graph": cdk.fingerprint.GraphOnlyFingerprinter(size, depth),
        "maccs": cdk.fingerprint.MACCSFingerprinter(),
        "pubchem": cdk.fingerprint.PubchemFingerprinter(cdk.silent.SilentChemObjectBuilder.getInstance()),
        "estate": cdk.fingerprint.EStateFingerprinter(),
        "hybridization": cdk.fingerprint.HybridizationFingerprinter(size, depth),
        "lingo": cdk.fingerprint.LingoFingerprinter(depth),
        "klekota-roth": cdk.fingerprint.KlekotaRothFingerprinter(),
        "shortestpath": cdk.fingerprint.ShortestPathFingerprinter(size),
        "signature": cdk.fingerprint.SignatureFingerprinter(depth),
        "circular": cdk.fingerprint.CircularFingerprinter(),
        "AtomPair": cdk.fingerprint.AtomPairs2DFingerprinter()
    }

    mol = cdk_parser_smiles(cdk, smi)
    if fp_type in _fingerprinters:
        fingerprinter = _fingerprinters[fp_type]
    else:
        raise IOError('invalid fingerprint type')

    fp = fingerprinter.getBitFingerprint(mol).asBitSet()
    bits = []
    idx = fp.nextSetBit(0)
    while idx >= 0:
        bits.append(idx)
        idx = fp.nextSetBit(idx + 1)
    return bits


def ob_fingerprint(smi, fp_type='FP2', nbit=307, output='bit'):
    mol = pybel.readstring("smi", smi)
    if fp_type == 'FP2':
        fp = mol.calcfp('FP2')
    elif fp_type == 'FP3':
        fp = mol.calcfp('FP3')
    elif fp_type == 'FP4':
        fp = mol.calcfp('FP4')
    bits = fp.bits
    bits = [x for x in bits if x < nbit]
    if output == 'bit':
        return bits
    else:
        vec = np.zeros(nbit)
        vec[bits] = 1
        vec = vec.astype(int)
        return vec


def get_fingerprint(cdk, smi, fp_type, nbit=None, depth=None):
    if fp_type in [
        "daylight", "extended", "graph",
        "pubchem", "estate", "hybridization",
        "lingo", "klekota-roth", "shortestpath",
        "signature", "circular", "AtomPair"
    ]:
        if nbit is None:
            nbit = 1024
        if depth is None:
            depth = 6
        res = cdk_fingerprint(cdk, smi, fp_type, nbit, depth)
    elif fp_type in [
        "rdkit", "maccs",
        "TopologicalTorsion", "Avalon"
    ]:
        res = rdk_fingerprint(smi, fp_type, nbit)
    elif fp_type in [
        "FP2", "FP3", "FP4"
    ]:
        if nbit is None:
            nbit = 307
        res = ob_fingerprint(smi, fp_type, nbit)
    else:
        raise IOError('invalid fingerprint type')
    return res


def convert_bitvect_to_numpy_array(fp, fp_type):
    if fp_type == 'maccs':
        nbit = 166
    elif fp_type == 'estate':
        nbit = 79
    elif fp_type == 'pubchem':
        nbit = 881
    elif fp_type == 'klekota-roth':
        nbit = 4860
    elif fp_type == 'AtomPair':
        nbit = 2048
    elif fp_type == ['FP2', 'FP3', 'FP4']:
        nbit = 307
    else:
        nbit = 2048

    bitvect = [0] * nbit
    for val in fp:
        bitvect[val - 1] = 1
    return np.array(list(bitvect))


def feature_engineer(df, column_name, prefix):
    
    features = []
    for row in tqdm(df.index, total=len(df.index)):
        s = df.loc[row, column_name]
        feature = []
        for char in s:
            feature.append(str(char))
        features.append(feature)
    
    data = pd.DataFrame(
        features,
        columns=[prefix + str(i) for i in range(len(feature))]
    )
    return data


class Fingerprint:
    def __init__(self, smi:pd.Series, save_dir, jvmpath = None, CDK_DIRPATH='misc'):
        os.makedirs(save_dir, exist_ok=True)
        # os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.181-7.b13.el7.x86_64/"
        CDK_PATH = os.path.join(CDK_DIRPATH, "PaDEL-Descriptor/lib/cdk-2.3.jar")
        
        if not isJVMStarted():
            jvmpath = jvmpath or getDefaultJVMPath()
            startJVM(jvmpath, "-ea", "-Djava.class.path=%s" % CDK_PATH)
        self.cdk = JPackage('org').openscience.cdk

        self.colnames=['SMILES_NS'] 
        
        self.df = pd.DataFrame(smi.tolist(),
                columns  = self.colnames)
        
        np.set_printoptions(threshold=sys.maxsize)

        self.all_fp_types = [
            "daylight", "extended", "graph", "pubchem",
            "estate", "hybridization", "lingo", "klekota-roth",
            "circular", "rdkit", "maccs", "Avalon",
            "FP2", "FP3", "FP4", "AtomPair"
        ]

        tqdm().pandas()

        for fp in self.all_fp_types:
            tmp = self.df.progress_apply(
                lambda row: convert_bitvect_to_numpy_array(
                    get_fingerprint(self.cdk, row['SMILES_NS'], fp_type=fp), fp), axis=1)
            self.df[fp] = tmp
        self.all_fp_types = [
            "daylight", "extended", "graph", "pubchem",
            "estate", "hybridization", "lingo", "klekota-roth",
            "circular", "rdkit", "maccs", "Avalon",
            "FP2", "FP3", "FP4", "AtomPair"
        ]

        tqdm().pandas()

        for fp in self.all_fp_types:
            self.df[fp] = self.df.progress_apply(
                lambda row: str(row[fp]).strip('[]'), axis=1)

        for col in self.df.columns:
            self.df[col] = self.df[col].astype(str).apply(lambda x: x.replace(" ", ""))
            self.df[col] = self.df[col].astype(str).apply(lambda x: x.replace("\n", ""))
        

        self.FP = self.df

        with open(save_dir + r'/FPtest.csv', "wb+") as FP_csv:
            self.FP.to_csv(save_dir + '/FPtest.csv', index=False)
        
        self.daylight = feature_engineer(
            df=self.FP,
            column_name="daylight",
            prefix="daylightFP"
        )
        self.extended = feature_engineer(
            df=self.FP,
            column_name="extended",
            prefix="extendedFP"
        )
        self.graph = feature_engineer(
            df=self.FP,
            column_name="graph",
            prefix="graphFP"
        )
        self.pubchem = feature_engineer(
            df=self.FP,
            column_name="pubchem",
            prefix="PubchemFP"
        )
        self.estate = feature_engineer(
            df=self.FP,
            column_name="estate",
            prefix="estateFP"
        )
        self.hybridization = feature_engineer(
            df=self.FP,
            column_name="hybridization",
            prefix="hybridizationFP"
        )
        self.lingo = feature_engineer(
            df=self.FP,
            column_name="lingo",
            prefix="lingoFP"
        )
        self.klekota_roth = feature_engineer(
            df=self.FP,
            column_name="klekota-roth",
            prefix="klekota_rothFP"
        )
        self.circular = feature_engineer(
            df=self.FP,
            column_name="circular",
            prefix="circularFP"
        )
        self.rdkit = feature_engineer(
            df=self.FP,
            column_name="rdkit",
            prefix="rdkitFP"
        )
        self.maccs = feature_engineer(
            df=self.FP,
            column_name="maccs",
            prefix="maccsFP"
        )
        self.Avalon = feature_engineer(
            df=self.FP,
            column_name="Avalon",
            prefix="AvalonFP"
        )
        self.AtomPair = feature_engineer(
            df=self.FP,
            column_name="AtomPair",
            prefix="AtomPairFP"
        )
        self.FP2 = feature_engineer(
            df=self.FP,
            column_name="FP2",
            prefix="FP2FP"
        )
        self.FP3 = feature_engineer(
            df=self.FP,
            column_name="FP3",
            prefix="FP3FP"
        )
        self.FP4 = feature_engineer(
            df=self.FP,
            column_name="FP4",
            prefix="FP4FP"
        )
            
        self.daylight.name = 'daylight'
        self.extended.name = 'extended'
        self.graph.name = 'graph'
        self.pubchem.name = 'pubchem'
        self.estate.name = 'estate'
        self.hybridization.name = 'hybridization'
        self.lingo.name = 'lingo'
        self.klekota_roth.name = 'klekota_roth'
        self.circular.name = 'circular'
        self.rdkit.name = 'rdkit'
        self.maccs.name = 'maccs'
        self.Avalon.name = 'Avalon'
        self.AtomPair.name = 'AtomPair'
        self.FP2.name = 'FP2'
        self.FP3.name = 'FP3'
        self.FP4.name = 'FP4'

        self.df_list = [
            self.daylight, self.extended, self.graph, self.pubchem,
            self.estate, self.hybridization, self.lingo, self.klekota_roth,
            self.circular, self.rdkit, self.maccs, self.AtomPair,
            self.Avalon, self.FP2, self.FP3, self.FP4
        ]

        # for df in self.df_list:
        #     with open(save_dir + r'/' + df.name + r'test.csv', "wb+") as fp_csv:
        #         df.to_csv(save_dir + '/' + df.name + r'test.csv', index=False)
        #     df = df.progress_apply(pd.to_numeric)

        self.FiLe1 = [self.estate, self.klekota_roth, self.lingo, self.maccs]

        for f in self.FiLe1:
            self.pubchem = pd.concat([self.pubchem, f], axis=1)

        self.FiLe2 = [
            self.Avalon, self.circular, self.daylight, self.extended, self.FP2,
            self.FP3, self.FP4, self.graph, self.hybridization, self.rdkit
        ]

        for f in self.FiLe2:
            self.AtomPair = pd.concat([self.AtomPair, f], axis=1)

        self.pubchem = self.pubchem[longlist.list_pubchem]  # list_pubchem is imported from longlist.py
        self.AtomPair = self.AtomPair[longlist.list_atompair]  # list_atompair is imported from longlist.py
        
        with open(save_dir + r'/fingerprint-hash.csv', "wb+") as hash_csv:
            self.AtomPair.to_csv(save_dir + '/fingerprint-hash.csv', index=False)
        with open(save_dir + r'/fingerprint-nonhash.csv', "wb+") as nonhash_csv:
            self.pubchem.to_csv(save_dir + r'/fingerprint-nonhash.csv', index=False)
