export default class QuestSystem {
  constructor(){ this.progress = {}; }
  addQuest(id, data){ this.progress[id] = data; }
}
