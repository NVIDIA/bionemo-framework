# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import pickle
import sqlite3

import tqdm


class HeterographStore:
    """sqlite3 database to store processed complex graphs from model training.
    The database table has ligand, and the complex graph.
    """

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS heterograph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ligand TEXT NOT NULL,
                complex_graph TEXT NOT NULL,
                rec_type int NULL
            );


            """
        )

    def __len__(self):
        return self.conn.execute("SELECT count(*) FROM heterograph").fetchone()[0]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = 0
            stop = len(self)

            if idx.start:
                start = idx.start
            if idx.stop:
                stop = idx.stop

            records = self.conn.execute(
                "SELECT ligand, complex_graph FROM heterograph LIMIT ?, ?", [start, stop - start]
            ).fetchall()

            complex_graphs = []
            for ligand, complex_graph in records:
                ligand = pickle.loads(ligand)
                complex_graph = pickle.loads(complex_graph)
                if ligand:
                    complex_graph.mol = ligand
                complex_graphs.append(complex_graph)
            return complex_graphs
        else:
            ligand, complex_graph = self.conn.execute(
                "SELECT ligand, complex_graph FROM heterograph LIMIT ?, 1", [idx]
            ).fetchone()
            ligand = pickle.loads(ligand)
            complex_graph = pickle.loads(complex_graph)
            if ligand:
                complex_graph.mol = ligand
            return complex_graph

    def insert(self, lig, complex_graph):
        self.conn.execute(
            "INSERT INTO heterograph(ligand, complex_graph) VALUES(?, ?)",
            [sqlite3.Binary(pickle.dumps(lig)), sqlite3.Binary(pickle.dumps(complex_graph))],
        )

    def commit(self):
        self.conn.commit()


class ConfidenceStore:
    """sqlite3 database to store ligand poses generated from reverse diffusion for confidence training.
    The database table has ligand names, ligand postions, and the RMSDs w.r.t the reference ligands
    """

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ligand_confidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ligand TEXT NOT NULL,
                ligand_position TEXT NOT NULL,
                rmsd TEXT NOT NULL
            );
            """
        )

    def __del__(self):
        self.conn.close()

    def __len__(self):
        return self.conn.execute("SELECT count(*) FROM ligand_confidence").fetchone()[0]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = 0
            stop = len(self)
            if idx.start:
                start = idx.start
            if idx.stop:
                stop = idx.stop
            records = self.conn.execute(
                "SELECT ligand, ligand_position, rmsd FROM ligand_confidence LIMIT ?, ?",
                [start, stop - start],
            ).fetchall()
            ligand_positions = []
            ligand_names = []
            ligand_rmsds = []
            for ligand, ligand_position, rmsd in records:
                ligand = pickle.loads(ligand)
                ligand_position = pickle.loads(ligand_position)
                ligand_rmsd = pickle.loads(rmsd)
                ligand_names.append(ligand)
                ligand_positions.append(ligand_position)
                ligand_rmsds.append(ligand_rmsd)
            return ligand_names, ligand_positions, ligand_rmsds
        else:
            ligand, ligand_position, ligand_rmsd = self.conn.execute(
                "SELECT ligand, ligand_position, rmsd FROM ligand_confidence LIMIT ?, 1",
                [idx],
            ).fetchone()
            lig = pickle.loads(ligand)
            ligand_position = pickle.loads(ligand_position)
            rmsd = pickle.loads(ligand_rmsd)
            return lig, ligand_position, rmsd

    def insert(self, lig, ligand_position, rmsd):
        data = (
            sqlite3.Binary(pickle.dumps(lig)),
            sqlite3.Binary(pickle.dumps(ligand_position)),
            sqlite3.Binary(pickle.dumps(rmsd)),
        )
        self.conn.execute(
            "INSERT INTO ligand_confidence(ligand, ligand_position, rmsd) VALUES(?, ?, ?)",
            data,
        )

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Split cached sqlite3 file, put each entry to a pickle file, which will later be converted to be web dataset compatible format"
    )

    parser.add_argument("sqlite_file", help="sqlite file or folder path with cached sqlite files inside")
    parser.add_argument(
        "--save_folder",
        help="folder path to save the pickle files. Optional",
        required=False,
        default=None,
    )

    args = parser.parse_args()

    sqlite_file = os.path.abspath(args.sqlite_file)
    if sqlite_file.endswith(".sqlite3"):
        sqlite_folder = os.path.abspath(os.path.join(sqlite_file, os.pardir))
        input_files = [os.path.basename(sqlite_file)]
        assert os.path.isfile(sqlite_file), f"{sqlite_file} is not file"
    else:
        sqlite_folder = sqlite_file
        input_files = []
        for file in os.listdir(sqlite_folder):
            if file.endswith(".sqlite3"):
                input_files.append(file)
                print("*" * 100)
                print(f"Found {file}")

        assert len(input_files) > 0, f"No sqlite3 found in {sqlite_folder}"

    save_folder = args.save_folder

    data_type = None
    if save_folder is None:
        if input_files[0] == "heterographs.sqlite3":
            foldername = os.path.basename(sqlite_folder)
            idx1, idx2 = foldername.index("INDEX"), foldername.index("maxLigSize")
            save_folder = os.path.abspath(
                os.path.join(sqlite_folder, os.pardir, foldername[:idx1] + foldername[idx2:])
            )
            data_type = "HeteroData"
        elif input_files[0] == "confidence_cache_id_base.sqlite3":
            foldername = os.path.basename(sqlite_folder)
            idx1, idx2 = foldername.index("split"), foldername.index("limit")
            save_folder = os.path.abspath(
                os.path.join(sqlite_folder, os.pardir, foldername[:idx1] + foldername[idx2:])
            )
            data_type = "LigandData"
        else:
            if "_heterographs.sqlite3" in input_files[0]:
                data_type = "HeteroData"
                save_folder = os.path.join(sqlite_folder, input_files[0].replace("_heterographs.sqlite3", ""))
            elif "_confidence_cache_id_base.sqlite3" in input_files[0]:
                data_type = "LigandData"
                save_folder = os.path.join(
                    sqlite_folder, input_files[0].replace("_confidence_cache_id_base.sqlite3", "")
                )

    print(f"Saving pickle files to {save_folder}")
    os.makedirs(save_folder, exist_ok=True)

    if data_type == "HeteroData":
        for file in input_files:
            sqlite_store = HeterographStore(os.path.join(sqlite_folder, file))
            num_samples = len(sqlite_store)
            for idx in tqdm.tqdm(range(num_samples)):
                complex_graph = sqlite_store[idx]
                pickle.dump(
                    complex_graph, open(os.path.join(save_folder, f"{complex_graph.name}.HeteroData.pyd"), "wb")
                )
    elif data_type == "LigandData":
        for file in input_files:
            sqlite_store = ConfidenceStore(os.path.join(sqlite_folder, file))
            num_samples = len(sqlite_store)
            for idx in tqdm.tqdm(range(num_samples)):
                name, lig_poses, lig_rmsds = sqlite_store[idx]
                pickle.dump(
                    [name, lig_poses, lig_rmsds], open(os.path.join(save_folder, f"{name}.LigandData.pyd"), "wb")
                )

    # print(sqlite_folder, args.save)


if __name__ == "__main__":
    main()
