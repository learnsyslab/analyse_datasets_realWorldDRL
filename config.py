from dataclasses import dataclass, field

@dataclass
class LegoSimpleDiffusion:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/stack_lego_simple_diffusion_deploy_1",
        "OliverHausdoerfer/stack_lego_simple_diffusion_deploy_3",
        "OliverHausdoerfer/stack_lego_simple_diffusion_deploy_4"
    ])

@dataclass
class LegoSimpleDitflow:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/stack_lego_simple_ditflow_deploy",
        "OliverHausdoerfer/stack_lego_simple_ditflow_deploy_1",
    ])

@dataclass
class LegoSimplePi05:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/stack_lego_simple_pi05_deploy_2",
        "OliverHausdoerfer/stack_lego_simple_pi05_deploy_1",
    ])


@dataclass
class SiemensDiffusion:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/siemens_diffusion_deploy_1",
    ])

@dataclass
class SiemensDitflow:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/siemens_ditflow_deploy_1",
    ])

@dataclass
class SiemensPi05:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/siemens_pi05_deploy_1",
    ])

# @dataclass
# class LegoSimpleGeneralizationDiffusion:
#     datasets: list = field(default_factory=lambda: [
#         "OliverHausdoerfer/stack_lego_simple_generalization_diffusion_deploy_1",
#         "OliverHausdoerfer/stack_lego_simple_generalization_diffusion_deploy_2",
#     ])

# @dataclass
# class LegoSimpleGeneralizationDitflow:
#     datasets: list = field(default_factory=lambda: [
#         "OliverHausdoerfer/stack_lego_simple_generalization_ditflow_deploy_2",
#         "OliverHausdoerfer/stack_lego_simple_generalization_ditflow_deploy_1",
#     ])

# @dataclass
# class LegoSimpleGeneralizationPi05:
#     datasets: list = field(default_factory=lambda: [
#         "OliverHausdoerfer/stack_lego_simple_pi05_generalization_deploy_2",
#         "OliverHausdoerfer/stack_lego_simple_pi05_generalization_deploy_1",
#     ])



