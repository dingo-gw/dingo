# How to use asimov with dingo

## GW150914 example

If we want to run a series of analyses on many different events, asimov is a
tool which will allow us to do this. To read more in depth about asimov, see
https://asimov.docs.ligo.org/asimov/master/getting-started.html. In this tutorial, 
we will go through how to analyze one gravitational wave event (GW150914) with DINGO 
through asimov. We will also discuss how to add more events to the ledger if we 
want to analyze a large number of events. 

The main principle behind asimov is that events are added to a ledger from which 
the settings can be read. There are three levels of settings to the ledger. First, 
is the "project settings" which will are common to all events which are being analyzed.
As an example, this could include the number of CPUs to use for the analysis.
Next, are the "event settings" which are common to all analyses of a certain event. For example,
this could be the trigger time of the analysis or the filepath to the calibration envelope.
Finally, we have the analysis settings which are specific to each inference run. For example,
this could be the network you want to use to analyze the event. 

The first step is to start a project which can be done with 

```
mkdir project_tutorial
cd project_tutorial
asimov init "init message"
```

If you type `ls -a` this should now show the following directory structure

```
asimov.log .asimov checkouts  logs  results  working
```

The asimov.log is where the log of all the asimov commands that were run for the 
project are stored. The `checkouts` folder will contain `.ini` files for the run. We will
see how to populate this folder with asimov in the next section. The other important folder 
is the `working` folder which will contain the results of the dingo runs. 

While the project has now been started, currently there are no settings applied to
the project. To apply some settings, download the `project_dingo.yaml` file from 
`https://github.com/dingo-gw/dingo/tree/main/examples/asimov` and apply them with:

```
asimov apply -f project_dingo.yaml
```

Now that some basic project settings have been added, we can start to analyze events.
Let's add the GW150914 event to the ledger. You can download the `event_dingo.yaml`
file from `https://github.com/dingo-gw/dingo/tree/main/examples/asimov` and apply it 
with

```
asimov apply -f ../../event_dingo.yaml
```

Finally, we can apply the analysis settings to GW150914. You can 
download a sample `analysis_dingo.yaml` file from
`https://github.com/dingo-gw/dingo/tree/main/examples/asimov`. 
You will need to change this yaml file to point to the location 
of your network .pt files. 

```
asimov apply -f ../../analysis_dingo.yaml -e GW150914_095045
```

We have now applied all the settings to GW150914 and are ready 
to begin the asimov run! Start by creating the .ini file using 

```
asimov manage build
```

This should create a .ini file in the `checkouts` subdirectory
which will be populated by the settings we applied above. Check 
that the trigger time, event name and outdir are the same as 
we specified. We can now submit the jobs. This has two steps. First,
the dag files are created using `dingo_pipe` based off the .ini 
files created using the `asimov manage build` command. Next the 
dag files will be submitted to the cluster. To do this run 

```
asimov manage submit
```

Now you should be all done! Check the output of the `working` directory
to ensure that log files are being created in accordance with 
starting a `dingo_pipe` run. You can run 

```
asimov monitor
```

to see the status of the runs. 


## Running on LIGO data

If you would like to run on LVK data, you will need to 
authenticate with your LVK credentials. This involves a
series of steps which are summarized here. First, 
you will need to install a few extra packages which 
are not shipped with default DINGO. To do this, 
run 

```
python -m pip install kerberos paramiko M2Crypto 
python -m pip install python-nds2-client
conda install python-ldas-tools-framecpp==2.6.14
```

Then, before running `asimov manage build` make sure to run 

```
export GWDATAFIND_SERVER=datafind.igwn.org
```

You will now need to authenticate with scitokens and 
store your credentials in the condor vault. To do this 
run the following commands and follow the prompts. 
They will ask for your LIGO credentials. 

```
export HTGETTOKENOPTS="-a vault.ligo.org -i igwn"
condor_vault_storer -v igwn
kinit 
htgettoken
```

Now you can run `condor_store_cred query-oauth` and you should 
see there are two credentials `igwn.top` and `igwn.use`. If 
you only see one credential, contact your sys admin or
upgrade to the latest version of condor. 