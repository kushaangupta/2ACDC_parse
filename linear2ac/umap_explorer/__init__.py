from .model import UmapExplorerModel
from .view import UmapExplorerView
from .controller import UmapExplorerController
class UmapExplorer():
    def __init__(self,umap,labels):
        self.controller = UmapExplorerController()
        self.model = UmapExplorerModel(umap,labels)
        self.view = UmapExplorerView(self.controller)
        # set references.
        self.model.controller = self.controller
        self.controller.model = self.model
        self.controller.view = self.view
        # set start command
        self.controller.update_display_data()
        self.controller.update_colormap()
        pass