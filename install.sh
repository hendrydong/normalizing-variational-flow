#!/bin/bash
# Copyright 2022 Statistics and Machine Learning Research Group at HKUST. All rights reserved.
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

conda create -n nvf python=3.8 -y
conda activate nvf
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
# conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt

wget https://zenodo.org/record/1161203/files/data.tar.gz?download=1
tar zxvf data.tar.gz?download=1
