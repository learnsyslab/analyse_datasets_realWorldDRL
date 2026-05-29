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
class SiemensDifficultPi05:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/siemens_difficult_pi05_deploy",
    ])

@dataclass
class SiemensDifficultPi05Generalization:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/siemens_difficult_generalization_pi05_deploy",
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

@dataclass
class OursLegoSimple:
    datasets: list = field(default_factory=lambda: [
        "gabormarko/franka-insert-lego-2x4-eval-normal-50",
    ])

@dataclass
class OursLegoFt20:
    datasets: list = field(default_factory=lambda: [
        "gabormarko/franka-insert-lego-2x4-eval-ft-20-v2",
    ])

@dataclass
class OursSiemens:
    datasets: list = field(default_factory=lambda: [
        "gabormarko/franka-insert-siemens-lid-eval-normal-50",
    ])

@dataclass
class OursSiemensGeneralization1:
    datasets: list = field(default_factory=lambda: [
        "gabormarko/franka-insert-siemens-lid-eval-ood-1-20",
    ])

# TODO should uncomment - just commented out because its slow for loading
# @dataclass
# class OursSiemensGeneralization2:
#     datasets: list = field(default_factory=lambda: [
#         "gabormarko/franka-insert-siemens-lid-eval-ood-2-20",
#     ])


@dataclass
class ShelfDiffusion:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/diffusion_shelf_deploy",
    ])

@dataclass
class ShelfDitflow:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/ditflow_shelf_deploy",
    ])

@dataclass
class ShelfDtiflowJim:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/ditflow_shelf_jim_deploy",
    ])

@dataclass
class ShelfPi05:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/pi05_shelf_deploy",
    ])

@dataclass
class ShelfPi05Generalization:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/pi05_shelf_deploy_ood",
    ])


@dataclass
class ShelfDiffusionGeneralization:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/diffusion_shelf_deploy_ood",
    ])

@dataclass
class ShelfDitflowGeneralization:
    datasets: list = field(default_factory=lambda: [
        "OliverHausdoerfer/ditflow_shelf_deploy_ood",
    ])









#######################




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

