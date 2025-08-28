# quick XML check (no simulator needed)
import xml.etree.ElementTree as ET

def dump_joint_axes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for j in root.findall(".//joint"):
        print(
            j.get("name"),
            j.get("type"),
            j.get("axis"),
            j.get("range"),
        )

# dump_joint_axes("assets/xml/universal_robots_ur10e/ur10e_robotiq.xml")
dump_joint_axes("assets/xml/universal_robots_ur10e/ur10e_2f85.xml")
