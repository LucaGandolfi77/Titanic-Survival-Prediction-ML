export class WebRTCAudio {
    constructor() {
        this.peerConnection = null;
        this.localStream = null;
        this.remoteStream = null;
        this.isReady = false;
        this.iceServers = [
            { urls: 'stun:stun.l.google.com:19302' },
            { urls: 'stun:stun1.l.google.com:19302' },
        ];
        this.onRemoteTrack = null;
        this.onConnectionState = null;
    }

    async init() {
        try {
            this.localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.isReady = true;
            return true;
        } catch {
            return false;
        }
    }

    createOffer() {
        if (!this.isReady || !this.localStream) return null;

        this.peerConnection = new RTCPeerConnection({ iceServers: this.iceServers });

        this.localStream.getTracks().forEach(track => {
            this.peerConnection.addTrack(track, this.localStream);
        });

        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate && this.onIceCandidate) {
                this.onIceCandidate(event.candidate);
            }
        };

        this.peerConnection.ontrack = (event) => {
            this.remoteStream = event.streams[0];
            if (this.onRemoteTrack) this.onRemoteTrack(this.remoteStream);
        };

        this.peerConnection.oniceconnectionstatechange = () => {
            if (this.onConnectionState) {
                this.onConnectionState(this.peerConnection.iceConnectionState);
            }
        };

        return this.peerConnection.createOffer();
    }

    async handleOffer(offer) {
        if (!this.isReady) {
            const ok = await this.init();
            if (!ok) return null;
        }

        this.peerConnection = new RTCPeerConnection({ iceServers: this.iceServers });

        this.localStream.getTracks().forEach(track => {
            this.peerConnection.addTrack(track, this.localStream);
        });

        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate && this.onIceCandidate) {
                this.onIceCandidate(event.candidate);
            }
        };

        this.peerConnection.ontrack = (event) => {
            this.remoteStream = event.streams[0];
            if (this.onRemoteTrack) this.onRemoteTrack(this.remoteStream);
        };

        this.peerConnection.oniceconnectionstatechange = () => {
            if (this.onConnectionState) {
                this.onConnectionState(this.peerConnection.iceConnectionState);
            }
        };

        await this.peerConnection.setRemoteDescription(new RTCSessionDescription(offer));
        const answer = await this.peerConnection.createAnswer();
        await this.peerConnection.setLocalDescription(answer);
        return answer;
    }

    async handleAnswer(answer) {
        if (!this.peerConnection) return;
        await this.peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
    }

    async addIceCandidate(candidate) {
        if (!this.peerConnection || !candidate) return;
        await this.peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
    }

    toggleMute() {
        if (!this.localStream) return;
        this.localStream.getAudioTracks().forEach(track => {
            track.enabled = !track.enabled;
        });
    }

    isMuted() {
        if (!this.localStream) return false;
        return !this.localStream.getAudioTracks()[0]?.enabled;
    }

    dispose() {
        if (this.localStream) {
            this.localStream.getTracks().forEach(t => t.stop());
        }
        if (this.peerConnection) {
            this.peerConnection.close();
        }
        this.peerConnection = null;
        this.localStream = null;
        this.remoteStream = null;
        this.isReady = false;
    }
}