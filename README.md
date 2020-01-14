# Firing Graph

This branch of the repository host the simulation presented in the arxiv paper [One the recovering of latent factore from sampling and firing graph](https://arxiv.org/abs/1909.09493) 
The simulations presented in the paper can be found in repository simulations. The dependencies of the project can be found in environement.yml and the virtual env can easily be built using conda. 
Finally the directory tests hosts unit tests.

#### Run simulations

 * Prepare for simulations
    
    ```
    git checkout publi_1
    cd $HOME/firing_graph
    conda env create -f environment.yml
    conda activate fg-env
    export PYTHONPATH="$HOME/firing_graph:$PYTHONPATH"
    python -m unittest discover tests
   ```

 * Run simulations

    ``` 
   python -m simulations.signal_plus_noise_1 &&
   python -m simulations.signal_plus_noise_2 &&
   python -m simulations.signal_plus_noise_3 &&
   python -m simulations.sparse_identification &&
   python -m simulations.sparse_identification_2
    ``` 

