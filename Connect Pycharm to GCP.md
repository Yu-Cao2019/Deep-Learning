# How to Connect Pycharm to GCP (Mac)

1. Open the terminal while the Xquartz is running. 
2. Type the following command in the terminal to connect to GCP:
```
cd ~/.ssh
ls
ssh -X -i gkey ubuntu@<External IP address from Dashbaord VMs>
```
3. Check capacity and other information by typing the following command:
```
nvidia-smi
nvcc -V
```
4. Open the Pycharm and create a new project or open the previous one.
5. Tools - deployment - configuration 
- Find the GCP created before or create a new one by click '+'
- Click SSH configuration and change the Host to the new external IP in VMs
- Clike 'Test Connection' to connect
- When it says 'connect successfully', you can move to 'Mappings', and set local path as your local path, deployment path as your path in ubuntu
6. PyCharm - Preference - Python interpreter 
- Click setting button on the right of Python Interpreter line, and show all
- Create a new interpreter with SSH Interpreter - Existing server configuration - gcp - add 3 after Interpreter line - change Sync folders to your path in ubuntu
