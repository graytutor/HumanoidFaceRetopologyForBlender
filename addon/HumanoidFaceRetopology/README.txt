HUMANOID FACE RETOPOLOGY (HFR)
==============================

Humanoid Face Retopology (HFR) is a Blender add-on for landmark-driven humanoid face retopology.
It uses a bundled symmetric face template, lets the user place facial/head/ear/neck landmarks on a high-poly target, then generates a lower-density retopology mesh fitted to the target surface.

Current version: v1.0.0

Development notice
------------------

This add-on was created using ChatGPT as a development assistant.
Project decisions, testing, packaging, and release preparation are maintained by the project owner.

Main workflow
-------------

1. Install and enable the add-on.
2. Open View3D > Sidebar > HFR.
3. Assign the high-poly object to Target Mesh.
4. Click Initialize HFR Workspace.
5. Click Add All Landmarks.
6. Move the landmarks to match the target face, head, ears, and neck.
7. Click Generate Retopology.
8. If the result needs correction, adjust landmarks and run Generate Retopology again.

The user is expected to prepare the target mesh before using HFR. The add-on assumes a face-oriented humanoid target with the front facing Blender world -Y, vertical direction on Z, and left/right across X.

Requirements
------------

- Blender 4.0+
- A humanoid high-poly target mesh
- The bundled template asset:
  - templates/HFRTemplate.blend
  - templates/HFRTemplateBinding.json

UI overview
-----------

1. Setup
--------

Used to assign the target mesh, load the bundled template, initialize the workspace, and check basic status.

Important status items:

- Target Mesh: target high-poly mesh is assigned.
- Template Mesh: bundled retopology template is loaded.
- Template Binding: template anchors are available.
- Landmarks: landmark objects are created.

2. Landmarks
------------

Used to create, reset, mirror, and display landmarks.

Recommended basic use:

- Click Add All Landmarks.
- Turn on Landmark Mirror X for symmetric targets.
- Move one side of the landmarks and let the other side follow.
- Use Landmark to Front when front landmarks are hard to see through the target mesh.

3. Generate
-----------

Used to create the final retopology output.

- Output Name controls the generated object name.
- Replace Existing removes previous HFR outputs before generating a new one.
- Wire Output displays the output as wire for inspection.
- Generate Retopology creates the fitted retopology mesh.

4. Cleanup
----------

Used to remove temporary HFR objects.

- Clean Unneeded Objects removes/moves stray HFR objects from the HFR collections.
- Delete HFR Landmarks / Guides removes landmark and guide objects.

Advanced and DevOption policy
-----------------------------

HFR separates non-basic controls into two layers.

Advanced
--------

Advanced is for users who want more control over final output and broad feature behavior.

Examples:

- Landmark save/load/export
- Target fitting options
- Output display options
- Snap strength and guard options
- Broad feature on/off controls, such as eye boundary, nose alar, ear local, and neck fit

DevOption
---------

DevOption is for development and template maintenance.

Examples:

- Template binding
- Anchor-group editing
- Binding validation reports
- Vertex diagnostics
- Low-level solver parameters
- Internal guide/style refresh tools

Release builds should hide DevOption by setting:

HFR_SHOW_DEV_OPTIONS = False

Development builds can keep it enabled:

HFR_SHOW_DEV_OPTIONS = True

Repository layout
-----------------

addon/HumanoidFaceRetopology/
  __init__.py
  README.txt
  templates/
    HFRTemplate.blend
    HFRTemplateBinding.json

docs/
  INSTALLATION.txt
  USER_GUIDE.txt
  ADVANCED_AND_DEVOPTIONS.txt
  DEVELOPER_GUIDE.txt
  RELEASE_CHECKLIST.txt
  TROUBLESHOOTING.txt

tools/
  build_release.py

Installation from a release zip
-------------------------------

1. Download the release zip, for example HumanoidFaceRetopology_v1_0_0_release.zip.
2. In Blender, open Edit > Preferences > Add-ons.
3. Click Install... and select the zip file.
4. Enable Humanoid Face Retopology(HFR).
5. Open View3D > Sidebar > HFR.

Development install
-------------------

For development, copy or symlink addon/HumanoidFaceRetopology into Blender's add-ons directory, then enable the add-on in Preferences.

License
-------

This project is distributed under the MIT License. See LICENSE.

Status
------

HFR v1.0.0 is the first public release candidate promoted from the v0.6.3 UI/UX cleanup build. The retopology generation, landmark deformation, template binding, and snap algorithms were not intentionally changed during the v1.0.0 promotion pass.
