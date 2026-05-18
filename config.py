from dataclasses import dataclass, field


@dataclass
class LegoSimpleDitflowNovice:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/lego_simple_jim_ditflow_deploy_1",
    ])

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
class LegoSimpleDiffusionGeneralization:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/stack_lego_simple_generalization_diffusion_deploy_1",
    ])

@dataclass
class LegoSimpleDitflowGeneralization:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/stack_lego_simple_generalization_ditflow_deploy_1",
    ])

@dataclass
class LegoSimplePi05Generalization:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/stack_lego_simple_pi05_generalization_deploy_1",
        "OliverHausdoerfer/stack_lego_simple_pi05_generalization_deploy_2",
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

@dataclass
class SiemensDifficultDiffusion:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/diffusion_siemens_difficult_deploy",
    ])

@dataclass
class SiemensDifficultDitflow:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/ditflow_siemens_difficult_deploy",
    ])


@dataclass
class SiemensDiffusionDiffiultGeneralization:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/diffusion_siemens_difficult_generalization_deploy",
    ])

@dataclass
class SiemensDitflowDiffiultGeneralization:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/ditflow_siemens_difficult_generalization_deploy",
    ])


@dataclass
class SiemensDiffusionGeneralization:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/siemens_diffusion_generalization_deploy_1",
    ])

@dataclass
class SiemensDitflowGeneralization:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/siemens_ditflow_generalization_deploy_1",
    ])

@dataclass
class SiemensPi05Generalization:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/siemens_pi05_generalization_deploy_1",
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



