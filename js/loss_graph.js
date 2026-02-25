import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "ACEStep.LossGraph",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Заметьте, мы теперь цепляемся к новой главной ноде ACEStepTrainer!
        if (nodeData.name === "ACEStepTrainer") {
            
            const originalComputeSize = nodeType.prototype.computeSize;
            nodeType.prototype.computeSize = function(out) {
                // Высота самой ноды (содержит всего пару переключателей и порты)
                let size = originalComputeSize ? originalComputeSize.apply(this, arguments) :[300, 100];
                
                // Запоминаем, где закончились порты и виджеты
                this.widgetsEndHeight = size[1]; 
                
                // Если картинка получена, увеличиваем ноду
                if (this.lossGraphImg && this.lossGraphImg.complete) {
                    const aspect = this.lossGraphImg.width / this.lossGraphImg.height;
                    const padding = 10;
                    
                    const graphWidth = Math.max(size[0], 300) - padding * 2;
                    const graphHeight = graphWidth / aspect;
                    
                    size[1] += graphHeight + padding * 2;
                }
                return size;
            };

            const originalOnDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function(ctx) {
                if (originalOnDrawForeground) {
                    originalOnDrawForeground.apply(this, arguments);
                }
                
                if (this.lossGraphImg && this.lossGraphImg.complete) {
                    const padding = 10;
                    const aspect = this.lossGraphImg.width / this.lossGraphImg.height;
                    
                    const graphWidth = this.size[0] - padding * 2;
                    const graphHeight = graphWidth / aspect;
                    
                    // График рисуется строго ниже последнего порта/виджета
                    const y = (this.widgetsEndHeight || 100) + padding;
                    
                    // Если холст попытался сжаться, раздвигаем его
                    if (this.size[1] < y + graphHeight + padding) {
                        this.size[1] = y + graphHeight + padding;
                    }

                    const x = padding;
                    
                    ctx.save();
                    // Фон
                    ctx.fillStyle = "#2b2b2b";
                    ctx.fillRect(x, y, graphWidth, graphHeight);
                    
                    // График
                    ctx.drawImage(this.lossGraphImg, x, y, graphWidth, graphHeight);
                    
                    // Рамка
                    ctx.strokeStyle = "#444444";
                    ctx.lineWidth = 1;
                    ctx.strokeRect(x, y, graphWidth, graphHeight);
                    ctx.restore();
                }
            };
        }
    },
    
    setup() {
        api.addEventListener("acestep_loss_update", (event) => {
            const data = event.detail;
            const node = app.graph._nodes.find((n) => n.id == data.node);
            
            if (node) {
                if (!node.lossGraphImg) {
                    node.lossGraphImg = new Image();
                }
                
                node.lossGraphImg.onload = () => {
                    const minSize = node.computeSize();
                    if (node.size[0] < minSize[0]) node.size[0] = minSize[0];
                    if (node.size[1] < minSize[1]) node.size[1] = minSize[1];
                    
                    node.setDirtyCanvas(true, true);
                };
                
                node.lossGraphImg.src = data.image;
            }
        });
    }
});