Model Development
=================

Input Data Preparation
----------------------

<Todo Placeholder section>

.. _estimation :

Estimation
----------

ActivitySim includes the ability to re-estimate submodels using choice model estimation
tools such as `larch <https://larch.driftless.xyz/>`__.  To do so, ActivitySim adopts
the concept of an estimation data bundle (EDB), which is a collection of the necessary
data to re-estimate a submodel.  For example, for the auto ownership submodel,
the EDB consists of the following files:

*   model settings - the auto_ownership_model_settings.yaml file

*   coefficients - the auto_ownership_coefficients.csv file with each coefficient
    name, value, and constrain set to True or False if the coefficient is estimatable

*   utilities specification - the auto_ownership_SPEC.csv utility expressions file

*   chooser and alternatives data - the auto_ownership_values_combined.csv file with
    all chooser and alternatives data such as household information, land use
    information, and the utility data components for each alternative

ActivitySim also includes Jupyter :ref:`estimation_example_notebooks` for estimating
submodels with larch, as well as an ``activitysim.estimation.larch`` submodule that
transforms EDBs into larch models.  Additional estimation software translators can
be added later if desired.

The combination of writing an EDB for a submodel + a larch estimation notebook
means users can easily re-estimate submodels. This combination of functionality means:

*   There is no duplication of model specifications. ActivitySim owns the specification
    and larch pivots off of it.  Users code model specifications and utility expressions
    in ActivitySim so as to facilitate ease of use and eliminate inconsistencies and
    errors between the code used to estimate the models and the code used to apply the
    models.

*   The EDB includes all the data and model structure information and the
    ``activitysim.estimation.larch`` submodule used by the example notebooks transforms
    the EDB to larch's data model for estimation.

*   Users are able to add zones, alternatives, new chooser data, new taz data, new
    modes, new coefficients, revise utilities, and revise nesting structures in
    ActivitySim and larch responds accordingly.

*   In the future ActivitySim may be more tightly coupled to Larch, but for now the
    integration is loosely coupled to provide flexibility. Users preferring a different
    estimation tool can write their own translator.

Workflow
~~~~~~~~

The general workflow for estimating models is shown in the following figures and
explained in more detail below.

.. image:: ../images/estimation_tools.jpg

* The user converts their household travel survey into ActivitySim format households, persons, tours, joint tour participants, and trip tables.  The households and persons tables must have the same fields as the synthetic population input tables since the surveyed households and persons will be run through the same set of submodels as the simulated households and persons.
* The ActivitySim estimation example ``scripts\infer.py`` module reads the ActivitySim format household travel survey files and checks for inconsistencies in the input tables versus the model design + calculates additional fields such as the household joint tour frequency based on the trips and joint tour participants table.  Survey households and persons observed choices much match the model design (i.e. a person cannot have more work tours than the model allows).
* ActivitySim is then run in estimation mode to read the ActivitySim format household travel survey files, run the ActivitySim submodels to write estimation data bundles (EDB) that contains the model utility specifications, coefficients, chooser data, and alternatives data for each submodel.  Estimation mode runs single-processed and without destination sampling.
* The relevant EDBs are read and transformed into the format required by the model estimation tool (i.e. larch) and then the coefficients are re-estimated. The ``activitysim.estimation.larch`` library is included for integration with larch and there is a Jupyter Notebook estimation example for each core submodel.  No changes to the model specification are made in the process.
* The user can then update the ActivitySim model coefficients file(s) for the estimated submodel and re-run the model in simulation mode.  The user may want to use the restartable pipeline feature of ActivitySim to just run the submodel of interest.

.. image:: ../images/estimation_example.jpg


.. _estimation_example:

Example
~~~~~~~

.. note::
   The estimation_mode.ipynb Jupyter :ref:`estimation_example_notebooks` also introduces estimation mode and walks the user through the process.

To run the estimation example, do the following:

* Activate the correct conda environment if needed
* Create a local copy of the estimation example folder

::

  activitysim create -e example_estimation_sf -d test_example_estimation_sf

* Run the example

::

  cd test_example_estimation_sf
  activitysim run -c configs_estimation/configs -c configs -o output -d data_sf


* ActivitySim should log some information and write outputs to the output folder, including EDBs for each submodel.  The estimation example runs for about 15 minutes and writes EDBs for 2000 households.
* Open :ref:`estimation_example_notebooks` for a specific submodel and then step through the notebook to re-estimate the sub-model.

The estimation example assumes the machine has sufficient RAM to run with chunking disabled (`chunk_training_mode: disabled`).  See :ref:`chunk_size` for more information.

Settings
~~~~~~~~

Additional settings for running ActivitySim in estimation mode are specified in the ``estimation.yaml`` file.  The settings are:

* ``enable`` - enable estimation, either True or False
* ``bundles`` - the list of submodels for which to write EDBs
* ``survey_tables`` - the list of input ActivitySim format survey tables with observed choices to override model simulation choices in order to write EDBs.  These tables are the output of the ``scripts\infer.py`` script that pre-processes the ActivitySim format household travel survey files for the example data and submodels


.. _estimation_example_notebooks:

Estimation Notebooks
~~~~~~~~~~~~~~~~~~~~

ActivitySim includes a `Jupyter Notebook <https://jupyter.org>`__ recipe book with interactive re-estimation examples for each estimatable submodel.  To run a Jupyter notebook, do the following:

* Open a conda prompt and activate the conda environment with ActivitySim installed
* If needed, ``conda install jupyterlab`` so you can run jupyter notebooks
* Type ``jupyter notebook`` to launch the web-based notebook manager
* Navigate to the ``examples/examples_estimaton/notebooks`` folder and select a notebook from the table below
* Save the updated coefficient file(s) to the configs folder and run the model in simulation mode

+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Example                             | Notebook                                                                                                                                                                            |
+=====================================+=====================================================================================================================================================================================+
| Estimation mode overview            | `01_estimation_mode.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/01_estimation_mode.ipynb>`_                       |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| School location                     | `02_school_location.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/02_school_location.ipynb>`_                       |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Work location                       | `03_work_location.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/03_work_location.ipynb>`_                           |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Auto ownership                      | `04_auto_ownership.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/04_auto_ownership.ipynb>`_                         |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Free parking                        | `05_free_parking.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/05_free_parking.ipynb>`_                             |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CDAP                                | `06_cdap.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/06_cdap.ipynb>`_                                             |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Mandatory tour frequency            | `07_mand_tour_freq.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/07_mand_tour_freq.ipynb>`_                         |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Work tour scheduling                | `08_work_tour_scheduling.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/08_work_tour_scheduling.ipynb>`_             |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| School tour scheduling              | `09_school_tour_scheduling.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/09_school_tour_scheduling.ipynb>`_         |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Joint tour frequency                | `10_joint_tour_freq.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/10_joint_tour_freq.ipynb>`_                       |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Jointatory tour composition         | `11_joint_tour_composition.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/11_joint_tour_composition.ipynb>`_         |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Jointatory tour participation       | `12_joint_tour_participation.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/12_joint_tour_participation.ipynb>`_     |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Joint nonmandatory tour destination | `13_joint_nonmand_tour_dest.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/13_joint_nonmand_tour_dest.ipynb>`_       |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Joint tour scheduling               | `14_joint_tour_scheduling.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/14_joint_tour_scheduling.ipynb>`_           |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Non mandatory tour frequency        | `15_non_mand_tour_freq.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/15_non_mand_tour_freq.ipynb>`_                 |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Non mandatory tour scheduling       | `16_nonmand_tour_scheduling.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/16_nonmand_tour_scheduling.ipynb>`_       |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Tour mode choice                    | `17_tour_mode_choice.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/17_tour_mode_choice.ipynb>`_                     |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Atwork subtour frequency            | `18_atwork_subtour_freq.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/18_atwork_subtour_freq.ipynb>`_               |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Atwork subtour destination          | `19_atwork_subtour_dest.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/19_atwork_subtour_dest.ipynb>`_               |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Atwork subtour scheduling           | `20_atwork_subtour_scheduling.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/20_atwork_subtour_scheduling.ipynb>`_   |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Stop frequency                      | `21_stop_frequency.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/21_stop_frequency.ipynb>`_                         |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Trip destination                    | `22_trip_dest.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/22_trip_dest.ipynb>`_                                   |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Trip mode choice                    | `23_trip_mode_choice.ipynb <https://github.com/activitysim/activitysim/blob/main/activitysim/examples/example_estimation/notebooks/23_trip_mode_choice.ipynb>`_                     |
+-------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


General API
~~~~~~~~~~~

.. automodule:: activitysim.estimation.larch.general
   :members:

.. automodule:: activitysim.estimation.larch.data_maker
   :members:

.. automodule:: activitysim.estimation.larch.simple_simulate
   :members:

Models API
~~~~~~~~~~

.. automodule:: activitysim.estimation.larch.auto_ownership
   :members:

.. automodule:: activitysim.estimation.larch.cdap
   :members:

.. automodule:: activitysim.estimation.larch.location_choice
   :members:

.. automodule:: activitysim.estimation.larch.mode_choice
   :members:

.. automodule:: activitysim.estimation.larch.nonmand_tour_freq
   :members:

.. automodule:: activitysim.estimation.larch.scheduling
   :members:

.. automodule:: activitysim.estimation.larch.stop_frequency
   :members:




Calibration
-----------

<Todo Placeholder>
